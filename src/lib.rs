// SPDX-License-Identifier: MIT

//! A human-friendly HID Report Descriptor Builder.
//!
//! The [hidreport] crate has another way of building a report descriptor,
//! see [hidreport::hid::ReportDescriptorBuilder].
//!
//! This crate focuses on producing HID Report Descriptors based on a descriptor of each report.
//! This is useful for e.g. reverse-engineered devices where one knows "the first two bytes are
//! x/y, then we have 8 bits for buttons and 1 byte for each wheel". 
//! To build a report one will thus create the individual fields and append them in-order to a
//! report, then build a HID Report Descriptor from all reports available.
//!
//! ## Fields
//!
//! HID Fields are the entities that represent data, e.g. "x" or "button 3". A field typically
//! has a [LogicalMinimum] and a [LogicalMaximum], optionally additionally a [PhysicalMinimum] and
//! [PhysicalMaximum]. Some fields may be specified with their [Unit] and [UnitExponent].
//!
//! Fields may be a single value ([VariableField]), an array of values ([ArrayField]) or used
//! for padding ([ConstantField]). It is common to pad out fields to the nearest byte where
//! required, e.g. a mouse with an [ArrayField] representing 3 buttons (via 3 bits) typically has 5
//! bits padding before the next field.
//!
//! ## Collections
//!
//! HID Collections are somewhat required (somewhat, not strictly) but they are very independent of
//! this report-based approach. Collections logically group fields together but they have
//! no influence on the data itself. For example, the x/y axes in the above example could be in
//! the same collection but may send data via to different HID Reports (identified by the
//! [ReportId]).
//!
//! For the construction of a HID Report Descriptor it is thus necessary to build the tree
//! of collections separately from the fields and assign each field to the respective collection.
//! Collections are nested, we call those without a parent collection a "top-level" collection.
//!
//! A field added to any collection is automatically a member of all of that collection's parent
//! collections.
//!
//! # Example
//!
//! This example shows how to build a HID Report Descriptor for a typical 3-button mouse.
//! ```
//! # use hidbuilder::{Collection, CollectionId, FieldBuilder, ReportBuilder, ReportDescriptorBuilder};
//! # use hidreport::{Usage, UsageMinimum, UsageMaximum, LogicalMinimum, LogicalMaximum, ReportId};
//! # use hut::{GenericDesktop, AsUsage};
//! #
//! # fn main() -> Result<(), hidbuilder::BuilderError> {
//! // Build a collection tree. We need at least one toplevel collection
//! // and then optionally nested collections within that. The most common setup
//! // is a top-level Application Collection that describe the use of the device
//! // (e.g. GenericDesktop::Mouse) and inside a Physical Collection that binds co-located features
//! // together (buttons + x/y + wheels).
//! //
//! // Collections aren't strictly needed and they're orthogonal to how fields are
//! // reported but oh well, it is what it is.
//!
//! // The tree is built with the top-level collection(s) first and other collections
//! // added to each respective parent collection.
//! let mut application: Collection = Collection::application(GenericDesktop::Mouse.usage().into());
//! let physical: CollectionId =
//!     Collection::physical(GenericDesktop::Pointer.usage().into())
//!         .add_to(&mut application);
//! // now convert the application collection into a toplevel collection
//! let application = application.toplevel();
//!
//! // Our first two data bytes are x and y, assigned to the physical collection.
//! // These two only differ in the usage, so we create a field with the common elements and clone
//! // that.
//! let axis_field = FieldBuilder::new(8.try_into().unwrap())
//!     .logical_range(&LogicalMinimum::from(-128), &LogicalMaximum::from(127))
//!     .collection(&physical)
//!     .variable();

//! let x = axis_field.clone()
//!     .usage(&GenericDesktop::X.usage().into())
//!     .build();
//! let y = axis_field.clone()
//!     .usage(&GenericDesktop::Y.usage().into())
//!     .build();
//!
//! // Array field for up to 3 simultaneous button presses
//! // Each element is 8 bits and can hold a button usage ID (1-10)
//! let buttons = FieldBuilder::new(8.try_into().unwrap())
//!     .logical_range(&LogicalMinimum::from(0), &LogicalMaximum::from(10))
//!     .collection(&physical)
//!     .array()
//!     .count(3.try_into().unwrap())?
//!     .usage_range(&UsageMinimum::from(0x00090001), &UsageMaximum::from(0x0009000A))
//!     .build();
//!
//! // Construct an input report from these fields with a report ID of 1
//! let input_report = ReportBuilder::new()
//!     .report_id(ReportId::from(1)) // this will be the first byte in the report
//!     .append_field(x) // second byte
//!     .append_field(y) // third byte
//!     .append_field(buttons) // bytes 4, 5, 6
//!     .input_report();
//!
//! // Build the HID report descriptor
//! let rdesc: Vec<u8> = ReportDescriptorBuilder::new()
//!     .add_toplevel_collection(application)
//!     .add_input_report(input_report)?
//!     .build()?;
//! # Ok(())
//! # }
//! ```

use hidreport::*;
use std::collections::BTreeMap;
use core::{
    num::NonZeroUsize,
    sync::atomic::{AtomicU32, Ordering},
};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BuilderError {
    #[error("Collection already added")]
    Collection,
    #[error("Report Descriptor size is invalid")]
    Size,
    #[error("Invalid Report Id")]
    ReportId,
}

type Result<T> = core::result::Result<T, BuilderError>;

#[doc(hidden)]
pub trait FieldBuilderState {}

macro_rules! impl_builder_state {
    ($s:ident) => {
        #[doc(hidden)]
        #[derive(Clone)]
        pub enum $s {}
        impl FieldBuilderState for $s {}
    };
}
impl_builder_state!(FieldBuilderNew);
impl_builder_state!(FieldBuilderVar);
impl_builder_state!(FieldBuilderVarWithUsage);
impl_builder_state!(FieldBuilderArr);
impl_builder_state!(FieldBuilderArrWithCount);
impl_builder_state!(FieldBuilderArrWithUsage);
impl_builder_state!(FieldBuilderArrWithCountAndUsage);
impl_builder_state!(FieldBuilderConst);

/// A builder for an individual Field within a report.
///
/// A field may be constant (i.e. padding), variable
/// or an array.
#[derive(Clone, Default)]
pub struct FieldBuilder<S: FieldBuilderState> {
    size_in_bits: usize,
    count: usize, // Array fields only
    usages: Vec<Usage>,
    usage_range: Option<(UsageMinimum, UsageMaximum)>,
    logical_minimum: Option<LogicalMinimum>,
    logical_maximum: Option<LogicalMaximum>,
    physical_minimum: Option<PhysicalMinimum>,
    physical_maximum: Option<PhysicalMaximum>,
    unit: Option<Unit>,
    exponent: Option<UnitExponent>,

    collection: Option<CollectionId>,
    // Reassure the compiler that S is used
    marker: core::marker::PhantomData<S>,
}

impl FieldBuilder<FieldBuilderNew> {
    /// Create a new field with the given size in bits
    pub fn new(size_in_bits: NonZeroUsize) -> FieldBuilder<FieldBuilderNew> {
        FieldBuilder {
            marker: core::marker::PhantomData,
            size_in_bits: size_in_bits.into(),
            count: 0,
            usages: Vec::new(),
            usage_range: None,
            logical_minimum: None,
            logical_maximum: None,
            physical_minimum: None,
            physical_maximum: None,
            unit: None,
            exponent: None,
            collection: None,
        }
    }

    /// Turn this field into a constant field
    pub fn constant(self) -> FieldBuilder<FieldBuilderConst> {
        FieldBuilder {
            size_in_bits: self.size_in_bits,
            count: 0,
            usages: Vec::new(),
            usage_range: None,
            logical_minimum: None,
            logical_maximum: None,
            physical_minimum: None,
            physical_maximum: None,
            unit: None,
            exponent: None,
            collection: self.collection,
            marker: core::marker::PhantomData,
        }
    }

    /// Turn this field into a variable field
    pub fn variable(self) -> FieldBuilder<FieldBuilderVar> {
        FieldBuilder {
            size_in_bits: self.size_in_bits,
            count: self.count,
            usages: Vec::new(),
            usage_range: None,
            logical_minimum: self.logical_minimum,
            logical_maximum: self.logical_maximum,
            physical_minimum: self.physical_minimum,
            physical_maximum: self.physical_maximum,
            unit: self.unit,
            exponent: self.exponent,
            collection: self.collection,
            marker: core::marker::PhantomData,
        }
    }

    /// Turn this field into an array field
    pub fn array(self) -> FieldBuilder<FieldBuilderArr> {
        FieldBuilder {
            size_in_bits: self.size_in_bits,
            count: self.count,
            usages: Vec::new(),
            usage_range: None,
            logical_minimum: self.logical_minimum,
            logical_maximum: self.logical_maximum,
            physical_minimum: self.physical_minimum,
            physical_maximum: self.physical_maximum,
            unit: self.unit,
            exponent: self.exponent,
            collection: self.collection,
            marker: core::marker::PhantomData,
        }
    }
}

impl<S: FieldBuilderState> FieldBuilder<S> {
    /// Set the Logical Range for this field
    ///
    /// A logical range with a positive [LogicalMinimum] is typically treated as
    /// an unsigned range.
    pub fn logical_range(mut self, min: &LogicalMinimum, max: &LogicalMaximum) -> Self {
        self.logical_minimum = Some(*min);
        self.logical_maximum = Some(*max);
        self
    }

    /// Set the Physical Range for this field
    ///
    /// A physical range with a positive [PhysicalMinimum] is typically treated as
    /// an unsigned range.
    pub fn physical_range(mut self, min: &PhysicalMinimum, max: &PhysicalMaximum) -> Self {
        self.physical_minimum = Some(*min);
        self.physical_maximum = Some(*max);
        self
    }

    /// Set the [Unit] for this field
    pub fn unit(mut self, unit: Unit) -> Self {
        self.unit = Some(unit);
        self
    }

    /// Set the [UnitExponent] for this field
    pub fn unit_exponent(mut self, exponent: UnitExponent) -> Self {
        self.exponent = Some(exponent);
        self
    }

    /// Assign this field to the given collection. The collection must
    /// be the deepest-nested collection that applies to the field.
    ///
    /// A [ConstantField] does not need to be assigned to a collection,
    /// these default to the current collection.
    pub fn collection(mut self, collection_id: &CollectionId) -> Self {
        self.collection = Some(collection_id.clone());
        self
    }
}

impl FieldBuilder<FieldBuilderVar> {
    /// Set the [Usage] for this field
    pub fn usage(self, usage: &Usage) -> FieldBuilder<FieldBuilderVarWithUsage> {
        FieldBuilder {
            marker: core::marker::PhantomData,
            size_in_bits: self.size_in_bits,
            count: self.count,
            usages: vec![*usage],
            usage_range: None,
            logical_minimum: self.logical_minimum,
            logical_maximum: self.logical_maximum,
            physical_minimum: self.physical_minimum,
            physical_maximum: self.physical_maximum,
            unit: self.unit,
            exponent: self.exponent,
            collection: self.collection,
        }
    }
}

impl FieldBuilder<FieldBuilderArr> {
    /// Set the set of [Usage] for this field
    pub fn usages(self, usages: &[Usage]) -> FieldBuilder<FieldBuilderArrWithUsage> {
        FieldBuilder {
            marker: core::marker::PhantomData,
            size_in_bits: self.size_in_bits,
            count: self.count,
            usages: Vec::from(usages),
            usage_range: None,
            logical_minimum: self.logical_minimum,
            logical_maximum: self.logical_maximum,
            physical_minimum: self.physical_minimum,
            physical_maximum: self.physical_maximum,
            unit: self.unit,
            exponent: self.exponent,
            collection: self.collection,
        }
    }

    /// Set the usage range 
    pub fn usage_range(
        self,
        usage_minimum: &UsageMinimum,
        usage_maximum: &UsageMaximum,
    ) -> FieldBuilder<FieldBuilderArrWithUsage> {
        FieldBuilder {
            marker: core::marker::PhantomData,
            size_in_bits: self.size_in_bits,
            count: self.count,
            usages: Vec::new(),
            usage_range: Some((*usage_minimum, *usage_maximum)),
            logical_minimum: self.logical_minimum,
            logical_maximum: self.logical_maximum,
            physical_minimum: self.physical_minimum,
            physical_maximum: self.physical_maximum,
            unit: self.unit,
            exponent: self.exponent,
            collection: self.collection,
        }
    }

    /// Set the number of elements in this array
    ///
    /// This validates that the field's size_in_bits is large enough to represent
    /// values in the logical range [logical_minimum, logical_maximum].
    pub fn count(self, count: NonZeroUsize) -> Result<FieldBuilder<FieldBuilderArrWithCount>> {
        // Calculate minimum bits needed to represent the logical range
        if let (Some(min), Some(max)) = (&self.logical_minimum, &self.logical_maximum) {
            let min_val: i32 = (*min).into();
            let max_val: i32 = (*max).into();
            let range = (max_val - min_val + 1).unsigned_abs() as u32;
            let bits_needed = if range <= 1 {
                1
            } else {
                32 - (range - 1).leading_zeros()
            };

            if self.size_in_bits < bits_needed as usize {
                return Err(BuilderError::Size);
            }
        }

        Ok(FieldBuilder {
            marker: core::marker::PhantomData,
            size_in_bits: self.size_in_bits,
            count: count.into(),
            usages: Vec::new(),
            usage_range: self.usage_range,
            logical_minimum: self.logical_minimum,
            logical_maximum: self.logical_maximum,
            physical_minimum: self.physical_minimum,
            physical_maximum: self.physical_maximum,
            unit: self.unit,
            exponent: self.exponent,
            collection: self.collection,
        })
    }
}

impl FieldBuilder<FieldBuilderArrWithCount> {
    #[doc(hidden)]
    pub fn usages(self, usages: &[Usage]) -> FieldBuilder<FieldBuilderArrWithCountAndUsage> {
        FieldBuilder {
            marker: core::marker::PhantomData,
            size_in_bits: self.size_in_bits,
            count: self.count,
            usages: Vec::from(usages),
            usage_range: None,
            logical_minimum: self.logical_minimum,
            logical_maximum: self.logical_maximum,
            physical_minimum: self.physical_minimum,
            physical_maximum: self.physical_maximum,
            unit: self.unit,
            exponent: self.exponent,
            collection: self.collection,
        }
    }

    #[doc(hidden)]
    pub fn usage_range(
        self,
        usage_minimum: &UsageMinimum,
        usage_maximum: &UsageMaximum,
    ) -> FieldBuilder<FieldBuilderArrWithCountAndUsage> {
        FieldBuilder {
            marker: core::marker::PhantomData,
            size_in_bits: self.size_in_bits,
            count: self.count,
            usages: Vec::new(),
            usage_range: Some((*usage_minimum, *usage_maximum)),
            logical_minimum: self.logical_minimum,
            logical_maximum: self.logical_maximum,
            physical_minimum: self.physical_minimum,
            physical_maximum: self.physical_maximum,
            unit: self.unit,
            exponent: self.exponent,
            collection: self.collection,
        }
    }
}

impl FieldBuilder<FieldBuilderArrWithUsage> {
    #[doc(hidden)]
    pub fn count(self, count: NonZeroUsize) -> Result<FieldBuilder<FieldBuilderArrWithCountAndUsage>> {
        // Calculate minimum bits needed to represent the logical range
        if let (Some(min), Some(max)) = (&self.logical_minimum, &self.logical_maximum) {
            let min_val: i32 = (*min).into();
            let max_val: i32 = (*max).into();
            let range = (max_val - min_val + 1).unsigned_abs() as u32;
            let bits_needed = if range <= 1 {
                1
            } else {
                32 - (range - 1).leading_zeros()
            };

            if self.size_in_bits < bits_needed as usize {
                return Err(BuilderError::Size);
            }
        }

        Ok(FieldBuilder {
            marker: core::marker::PhantomData,
            size_in_bits: self.size_in_bits,
            count: count.into(),
            usages: Vec::new(),
            usage_range: self.usage_range,
            logical_minimum: self.logical_minimum,
            logical_maximum: self.logical_maximum,
            physical_minimum: self.physical_minimum,
            physical_maximum: self.physical_maximum,
            unit: self.unit,
            exponent: self.exponent,
            collection: self.collection,
        })
    }
}

impl FieldBuilder<FieldBuilderArrWithCountAndUsage> {
    pub fn build(self) -> Field {
        Field::Array(ArrayField {
            size_in_bits: self.size_in_bits,
            count: self.count,
            usages: self.usages,
            usage_range: self.usage_range,
            logical_minimum: self.logical_minimum,
            logical_maximum: self.logical_maximum,
            physical_minimum: self.physical_minimum,
            physical_maximum: self.physical_maximum,
            unit: self.unit,
            exponent: self.exponent,
            collection: self.collection,
        })
    }
}

impl FieldBuilder<FieldBuilderVarWithUsage> {
    #[doc(hidden)]
    pub fn build(self) -> Field {
        Field::Variable(VariableField {
            size_in_bits: self.size_in_bits,
            usage: self.usages.first().unwrap().clone(),
            logical_minimum: self.logical_minimum,
            logical_maximum: self.logical_maximum,
            physical_minimum: self.physical_minimum,
            physical_maximum: self.physical_maximum,
            unit: self.unit,
            exponent: self.exponent,
            collection: self.collection,
        })
    }
}

impl FieldBuilder<FieldBuilderConst> {
    #[doc(hidden)]
    pub fn build(self) -> Field {
        Field::Constant(ConstantField {
            size_in_bits: self.size_in_bits,
            collection: self.collection,
        })
    }
}

/// A completed variable field
#[derive(Clone)]
pub struct VariableField {
    size_in_bits: usize,
    usage: Usage,
    logical_minimum: Option<LogicalMinimum>,
    logical_maximum: Option<LogicalMaximum>,
    physical_minimum: Option<PhysicalMinimum>,
    physical_maximum: Option<PhysicalMaximum>,
    unit: Option<Unit>,
    exponent: Option<UnitExponent>,
    collection: Option<CollectionId>,
}

/// A completed array field
#[derive(Clone)]
pub struct ArrayField {
    size_in_bits: usize,
    count: usize, // Array fields only
    usages: Vec<Usage>,
    usage_range: Option<(UsageMinimum, UsageMaximum)>,
    logical_minimum: Option<LogicalMinimum>,
    logical_maximum: Option<LogicalMaximum>,
    physical_minimum: Option<PhysicalMinimum>,
    physical_maximum: Option<PhysicalMaximum>,
    unit: Option<Unit>,
    exponent: Option<UnitExponent>,
    collection: Option<CollectionId>,
}

/// A completed constant field
#[derive(Clone)]
pub struct ConstantField {
    size_in_bits: usize,
    collection: Option<CollectionId>,
}

/// A completed field
#[derive(Clone)]
pub enum Field {
    Variable(VariableField),
    Array(ArrayField),
    Constant(ConstantField),
}

impl Field {
    /// The size of this fields in bits
    fn size_in_bits(&self) -> usize {
        match self {
            Field::Variable(f) => f.size_in_bits,
            Field::Array(f) => f.size_in_bits,
            Field::Constant(f) => f.size_in_bits,
        }
    }

    /// Get the collection this field belongs to
    pub fn collection(&self) -> Option<&CollectionId> {
        match self {
            Field::Variable(f) => f.collection.as_ref(),
            Field::Array(f) => f.collection.as_ref(),
            Field::Constant(f) => f.collection.as_ref(),
        }
    }
}

static COLLECTION_ID_COUNTER: AtomicU32 = AtomicU32::new(0);

/// Unique identifier for a collection
#[derive(Clone, Debug, PartialEq, PartialOrd, Ord, Eq)]
pub struct CollectionId {
    id: u32,
}

/// An application collection to be added to a [Field]
///
/// Typically a HID Report Descriptor has one or more Application
/// Collections that determine the use of this device.
/// In the Linux kernel, a HID device with multiple Application Collections
/// typically gets one `/dev/input/event` node for each collection.
pub struct ApplicationCollection {
    id: CollectionId,
    usage: Usage,
    children: Vec<Collection>,
    is_toplevel: bool,
}

/// An physical collection to be added to a [Field].
///
/// Physical Collections group elements of a device that
/// are physically co-located, e.g. the buttons, x/y axes and wheel
/// of a mouse are usually within one physical collection.
///
/// Nested Physical Collections further group elements together
/// where appropriate (e.g. buttons situatated on the same side of a
/// joystick).
pub struct PhysicalCollection {
    id: CollectionId,
    usage: Usage,
    children: Vec<Collection>,
    is_toplevel: bool,
}

/// A logical collection to be added to a [Field]
pub struct LogicalCollection {
    id: CollectionId,
    usage: Usage,
    children: Vec<Collection>,
    is_toplevel: bool,
}

/// A collection to be added to a [Field]
///
/// While a field does not have to be a member of a collection
/// it is strongly recommended to match the behavior of most
/// HID Report Descriptors from other devices.
pub enum Collection {
    Application(ApplicationCollection),
    Physical(PhysicalCollection),
    Logical(LogicalCollection),
}

/// A top-level collection that can be added to a [ReportDescriptorBuilder]
///
/// Top-level collections are those that have no parent collection. They are typically
/// Application Collections but can also be Physical or Logical Collections.
///
/// Use [Collection::toplevel()] to convert a [Collection] to a [ToplevelCollection].
pub struct ToplevelCollection {
    collection: Collection,
}

impl ToplevelCollection {
    /// Get the unique identifier for this collection
    pub fn id(&self) -> CollectionId {
        self.collection.id()
    }

    /// Consume this ToplevelCollection and return the inner Collection
    pub(crate) fn into_inner(self) -> Collection {
        self.collection
    }
}

impl Collection {
    /// Create a new Application Collection with the given usage
    pub fn application(usage: Usage) -> Self {
        Collection::Application(ApplicationCollection {
            id: CollectionId {
                id: COLLECTION_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            },
            children: Vec::new(),
            usage: usage,
            is_toplevel: true,
        })
    }

    /// Create a new Physical Collection with the given usage
    pub fn physical(usage: Usage) -> Self {
        Collection::Physical(PhysicalCollection {
            id: CollectionId {
                id: COLLECTION_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            },
            children: Vec::new(),
            usage: usage,
            is_toplevel: true,
        })
    }

    /// Create a new Logical Collection with the given usage
    pub fn logical(usage: Usage) -> Self {
        Collection::Logical(LogicalCollection {
            id: CollectionId {
                id: COLLECTION_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            },
            children: Vec::new(),
            usage: usage,
            is_toplevel: true,
        })
    }

    pub(crate) fn id(&self) -> CollectionId {
        match self {
            Collection::Application(c) => c.id.clone(),
            Collection::Physical(c) => c.id.clone(),
            Collection::Logical(c) => c.id.clone(),
        }
    }

    pub(crate) fn usage(&self) -> &Usage {
        match self {
            Collection::Application(c) => &c.usage,
            Collection::Physical(c) => &c.usage,
            Collection::Logical(c) => &c.usage,
        }
    }

    pub(crate) fn children(&self) -> &[Collection] {
        match self {
            Collection::Application(c) => &c.children,
            Collection::Physical(c) => &c.children,
            Collection::Logical(c) => &c.children,
        }
    }

    /// Add this collection to the given parent collection, returning the
    /// unique [CollectionId] for this collection.
    ///
    /// This [CollectionId] can then be used in [FieldBuilder::collection()].
    pub fn add_to(mut self, parent: &mut Collection) -> CollectionId {
        let id = self.id();

        // Mark the child as no longer toplevel since it's being added to a parent
        match &mut self {
            Collection::Application(c) => c.is_toplevel = false,
            Collection::Physical(c) => c.is_toplevel = false,
            Collection::Logical(c) => c.is_toplevel = false,
        }

        // Add the child to the parent's children
        match parent {
            Collection::Application(c) => c.children.push(self),
            Collection::Physical(c) => c.children.push(self),
            Collection::Logical(c) => c.children.push(self),
        };

        id
    }

    /// Convert this collection into a [ToplevelCollection] that can be added to a [ReportDescriptorBuilder]
    ///
    /// This consumes the collection and returns a [ToplevelCollection] wrapper.
    /// The collection must be a top-level collection (i.e., it must not have been added to a parent
    /// collection via [Collection::add_to()]).
    pub fn toplevel(self) -> ToplevelCollection {
        ToplevelCollection {
            collection: self,
        }
    }
}

#[doc(hidden)]
pub trait ReportIdState {}
macro_rules! impl_report_id_state {
    ($s:ident) => {
        #[doc(hidden)]
        #[derive(Clone)]
        pub enum $s {}
        impl ReportIdState for $s {}
    };
}
impl_report_id_state!(WithReportId);
impl_report_id_state!(WithoutReportId);

#[doc(hidden)]
pub trait ReportDescriptorBuilderState {}
macro_rules! impl_report_desc_builder_state {
    ($s:ident) => {
        #[doc(hidden)]
        #[derive(Clone)]
        pub enum $s {}
        impl ReportDescriptorBuilderState for $s {}
    };
}
impl_report_desc_builder_state!(New);
impl_report_desc_builder_state!(BuilderWithReportId);
impl_report_desc_builder_state!(BuilderWithoutReportIdIOF);
impl_report_desc_builder_state!(BuilderWithoutReportIdIxx);
impl_report_desc_builder_state!(BuilderWithoutReportIdIOx);
impl_report_desc_builder_state!(BuilderWithoutReportIdIxF);
impl_report_desc_builder_state!(BuilderWithoutReportIdxOx);
impl_report_desc_builder_state!(BuilderWithoutReportIdxOF);
impl_report_desc_builder_state!(BuilderWithoutReportIdxxF);

/// A report builder to build one Input Report, Output Report or Feature report.
///
/// The report must be built in the order of the fields appearing in the report.
#[derive(Clone)]
pub struct ReportBuilder<S: ReportIdState> {
    report_id: Option<ReportId>,
    fields: Vec<Field>,

    // Reassure the compiler that S is used
    marker: core::marker::PhantomData<S>,
}

impl ReportBuilder<WithoutReportId> {
    pub fn new() -> ReportBuilder<WithoutReportId> {
        ReportBuilder {
            marker: core::marker::PhantomData,
            report_id: None,
            fields: Vec::new(),
        }
    }

    /// Set the Report Id for this report
    ///
    /// Adding multiple reports of the same type with the same
    /// Report Id to a HID Report Descriptor is invalid.
    /// It is permitted to have the same Report Id for reports
    /// of different types, e.g. one input report and one output report.
    pub fn report_id(self, report_id: ReportId) -> ReportBuilder<WithReportId> {
        ReportBuilder {
            marker: core::marker::PhantomData,
            report_id: Some(report_id),
            fields: self.fields,
        }
    }

    /// Combine the current fields into an input report
    pub fn input_report(self) -> InputReport<WithoutReportId> {
        InputReport {
            report_id: self.report_id,
            fields: self.fields,
            marker: core::marker::PhantomData,
        }
    }

    /// Combine the current fields into an output report
    pub fn output_report(self) -> OutputReport<WithoutReportId> {
        OutputReport {
            report_id: self.report_id,
            fields: self.fields,
            marker: core::marker::PhantomData,
        }
    }

    /// Combine the current fields into a feature report
    pub fn feature_report(self) -> FeatureReport<WithoutReportId> {
        FeatureReport {
            report_id: self.report_id,
            fields: self.fields,
            marker: core::marker::PhantomData,
        }
    }
}

impl ReportBuilder<WithReportId> {
    /// Combine the current fields into an input report
    pub fn input_report(self) -> InputReport<WithReportId> {
        InputReport {
            report_id: self.report_id,
            fields: self.fields,
            marker: core::marker::PhantomData,
        }
    }

    /// Combine the current fields into an output report
    pub fn output_report(self) -> OutputReport<WithReportId> {
        OutputReport {
            report_id: self.report_id,
            fields: self.fields,
            marker: core::marker::PhantomData,
        }
    }

    /// Combine the current fields into a feature report
    pub fn feature_report(self) -> FeatureReport<WithReportId> {
        FeatureReport {
            report_id: self.report_id,
            fields: self.fields,
            marker: core::marker::PhantomData,
        }
    }
}

impl<S: ReportIdState> ReportBuilder<S> {
    /// Add the field at the current position in the report.
    ///
    /// The first field added is always at byte 0 for reports without a Report Id
    /// and at byte 1 for reports with a Report Id. Subsequent fields are added
    /// at the position indicated by the bit-size of all fields prior.
    pub fn append_field(mut self, field: Field) -> Self {
        self.fields.push(field);
        self
    }

    fn pad_to(mut self, size: usize) -> Self {
        let bits = self.fields.iter().fold(0, |acc, f| acc + f.size_in_bits());
        let pad_size = size - bits % size;
        if pad_size > 0 {
            let pad = FieldBuilder::new(pad_size.try_into().unwrap());
            self.fields.push(pad.constant().build());
        }
        self
    }

    /// Pad this report to the nearest 8-bit boundary (i.e. insert a [ConstantField] of
    /// the right size).
    pub fn pad_to_8(self) -> Self {
        self.pad_to(8)
    }

    /// Pad this report to the nearest 16-bit boundary (i.e. insert a [ConstantField] of
    /// the right size).
    pub fn pad_to_16(self) -> Self {
        self.pad_to(16)
    }

    /// Pad this report to the nearest 32-bit boundary (i.e. insert a [ConstantField] of
    /// the right size).
    pub fn pad_to_32(self) -> Self {
        self.pad_to(24)
    }
}

impl ReportBuilder<WithoutReportId> {}

#[derive(Clone)]
pub struct InputReport<S: ReportIdState> {
    report_id: Option<ReportId>,
    fields: Vec<Field>,

    marker: core::marker::PhantomData<S>,
}

#[derive(Clone)]
pub struct OutputReport<S: ReportIdState> {
    report_id: Option<ReportId>,
    fields: Vec<Field>,

    marker: core::marker::PhantomData<S>,
}

#[derive(Clone)]
pub struct FeatureReport<S: ReportIdState> {
    report_id: Option<ReportId>,
    fields: Vec<Field>,

    marker: core::marker::PhantomData<S>,
}

pub struct ReportDescriptorBuilder<S: ReportDescriptorBuilderState> {
    input_reports: Option<Vec<InputReport<WithReportId>>>,
    output_reports: Option<Vec<OutputReport<WithReportId>>>,
    feature_reports: Option<Vec<FeatureReport<WithReportId>>>,

    input_report: Option<InputReport<WithoutReportId>>,
    output_report: Option<OutputReport<WithoutReportId>>,
    feature_report: Option<FeatureReport<WithoutReportId>>,

    toplevel_collections: Vec<Collection>,

    marker: core::marker::PhantomData<S>,
}

/// Type of HID report
#[derive(Clone, Copy)]
enum ReportType {
    Input,
    Output,
    Feature,
}

/// State tracker for HID descriptor generation
/// Tracks current global items to avoid redundant emissions
struct DescriptorStack {
    usage_page: Option<UsagePage>,
    logical_minimum: Option<LogicalMinimum>,
    logical_maximum: Option<LogicalMaximum>,
    physical_minimum: Option<PhysicalMinimum>,
    physical_maximum: Option<PhysicalMaximum>,
    unit: Option<Unit>,
    unit_exponent: Option<UnitExponent>,
}

impl DescriptorStack {
    fn new() -> Self {
        Self {
            usage_page: None,
            logical_minimum: None,
            logical_maximum: None,
            physical_minimum: None,
            physical_maximum: None,
            unit: None,
            unit_exponent: None,
        }
    }

    /// Emit a usage page if it differs from the current state
    fn emit_usage_page(&mut self, bytes: &mut Vec<u8>, page: &UsagePage) {
        use hidreport::hid::{GlobalItem, ItemType};

        if self.usage_page.as_ref() != Some(page) {
            self.usage_page = Some(*page);
            bytes.extend_from_slice(&ItemType::Global(GlobalItem::UsagePage(*page)).as_bytes());
        }
    }

    /// Emit a usage id. Re-used usage-ids aren't a thing so this unconditionally
    /// emits the usage id, it's just for more coherence with the API.
    fn emit_usage_id(&mut self, bytes: &mut Vec<u8>, usage_id: &UsageId) {
        use hidreport::hid::{ItemType, LocalItem};
        bytes.extend_from_slice(&ItemType::Local(LocalItem::UsageId(*usage_id)).as_bytes());
    }

    /// Emit logical range if it differs from current state
    fn emit_logical_range(
        &mut self,
        bytes: &mut Vec<u8>,
        min: Option<LogicalMinimum>,
        max: Option<LogicalMaximum>,
    ) {
        use hidreport::hid::{GlobalItem, ItemType};

        if let Some(minimum) = min {
            if self.logical_minimum.as_ref() != Some(&minimum) {
                self.logical_minimum = Some(minimum);
                bytes.extend_from_slice(
                    &ItemType::Global(GlobalItem::LogicalMinimum(minimum)).as_bytes(),
                );
            }
        }
        if let Some(maximum) = max {
            if self.logical_maximum.as_ref() != Some(&maximum) {
                self.logical_maximum = Some(maximum);
                bytes.extend_from_slice(
                    &ItemType::Global(GlobalItem::LogicalMaximum(maximum)).as_bytes(),
                );
            }
        }
    }

    /// Emit physical range if it differs from current state
    fn emit_physical_range(
        &mut self,
        bytes: &mut Vec<u8>,
        min: Option<PhysicalMinimum>,
        max: Option<PhysicalMaximum>,
    ) {
        use hidreport::hid::{GlobalItem, ItemType};

        if let Some(minimum) = min {
            if self.physical_minimum.as_ref() != Some(&minimum) {
                self.physical_minimum = Some(minimum);
                bytes.extend_from_slice(
                    &ItemType::Global(GlobalItem::PhysicalMinimum(minimum)).as_bytes(),
                );
            }
        }
        if let Some(maximum) = max {
            if self.physical_maximum.as_ref() != Some(&maximum) {
                self.physical_maximum = Some(maximum);
                bytes.extend_from_slice(
                    &ItemType::Global(GlobalItem::PhysicalMaximum(maximum)).as_bytes(),
                );
            }
        }
    }

    /// Emit unit if it differs from current state
    fn emit_unit(&mut self, bytes: &mut Vec<u8>, unit: Option<Unit>) {
        use hidreport::hid::{GlobalItem, ItemType};

        if let Some(u) = unit {
            if self.unit.as_ref() != Some(&u) {
                self.unit = Some(u);
                bytes.extend_from_slice(&ItemType::Global(GlobalItem::Unit(u)).as_bytes());
            }
        }
    }

    /// Emit unit exponent if it differs from current state
    fn emit_unit_exponent(&mut self, bytes: &mut Vec<u8>, exponent: Option<UnitExponent>) {
        use hidreport::hid::{GlobalItem, ItemType};

        if let Some(exp) = exponent {
            if self.unit_exponent.as_ref() != Some(&exp) {
                self.unit_exponent = Some(exp);
                bytes
                    .extend_from_slice(&ItemType::Global(GlobalItem::UnitExponent(exp)).as_bytes());
            }
        }
    }
}

impl ReportDescriptorBuilder<New> {
    /// Create a new report descriptor builder.
    ///
    /// This builder may have reports with a [ReportId] added via
    /// [ReportDescriptorBuilder::add_input_report],
    /// [ReportDescriptorBuilder::add_output_report], and
    /// [ReportDescriptorBuilder::add_feature_report].
    ///
    /// Alternatively and if the reports do not contain a [ReportId], one of each report type may
    /// be set with
    /// [ReportDescriptorBuilder::set_input_report],
    /// [ReportDescriptorBuilder::set_output_report], and
    /// [ReportDescriptorBuilder::set_feature_report].
    pub fn new() -> ReportDescriptorBuilder<New> {
        ReportDescriptorBuilder {
            marker: core::marker::PhantomData,
            input_reports: None,
            output_reports: None,
            feature_reports: None,
            input_report: None,
            output_report: None,
            feature_report: None,
            toplevel_collections: Vec::new(),
        }
    }
}

impl ReportDescriptorBuilder<BuilderWithReportId> {
    #[doc(hidden)]
    pub fn add_input_report(
        mut self,
        input_report: InputReport<WithReportId>,
    ) -> Result<ReportDescriptorBuilder<BuilderWithReportId>> {
        let reports = self.input_reports.get_or_insert_with(Vec::new);
        // Check for duplicate ReportId
        if let Some(rid) = input_report.report_id {
            if reports.iter().any(|r| r.report_id == Some(rid)) {
                return Err(BuilderError::ReportId);
            }
        }
        reports.push(input_report);
        Ok(self)
    }

    #[doc(hidden)]
    pub fn add_output_report(
        mut self,
        output_report: OutputReport<WithReportId>,
    ) -> Result<ReportDescriptorBuilder<BuilderWithReportId>> {
        let reports = self.output_reports.get_or_insert_with(Vec::new);
        // Check for duplicate ReportId
        if let Some(rid) = output_report.report_id {
            if reports.iter().any(|r| r.report_id == Some(rid)) {
                return Err(BuilderError::ReportId);
            }
        }
        reports.push(output_report);
        Ok(self)
    }

    #[doc(hidden)]
    pub fn add_feature_report(
        mut self,
        feature_report: FeatureReport<WithReportId>,
    ) -> Result<ReportDescriptorBuilder<BuilderWithReportId>> {
        let reports = self.feature_reports.get_or_insert_with(Vec::new);
        // Check for duplicate ReportId
        if let Some(rid) = feature_report.report_id {
            if reports.iter().any(|r| r.report_id == Some(rid)) {
                return Err(BuilderError::ReportId);
            }
        }
        reports.push(feature_report);
        Ok(self)
    }
}

impl ReportDescriptorBuilder<New> {
    /// Set the input report for this builder.
    ///
    /// Reports without a [ReportId] may only have one report of the same type.
    pub fn set_input_report(
        self,
        input_report: InputReport<WithoutReportId>,
    ) -> Result<ReportDescriptorBuilder<BuilderWithoutReportIdIxx>> {
        Ok(ReportDescriptorBuilder {
            marker: core::marker::PhantomData,
            input_reports: None,
            output_reports: None,
            feature_reports: None,
            input_report: Some(input_report),
            output_report: None,
            feature_report: None,
            toplevel_collections: self.toplevel_collections,
        })
    }

    /// Set the output report for this builder.
    ///
    /// Reports without a [ReportId] may only have one report of the same type.
    pub fn set_output_report(
        self,
        output_report: OutputReport<WithoutReportId>,
    ) -> Result<ReportDescriptorBuilder<BuilderWithoutReportIdxOx>> {
        Ok(ReportDescriptorBuilder {
            marker: core::marker::PhantomData,
            input_reports: None,
            output_reports: None,
            feature_reports: None,
            input_report: None,
            output_report: Some(output_report),
            feature_report: None,
            toplevel_collections: self.toplevel_collections,
        })
    }

    /// Set the feature report for this builder.
    ///
    /// Reports without a [ReportId] may only have one report of the same type.
    pub fn set_feature_report(
        self,
        feature_report: FeatureReport<WithoutReportId>,
    ) -> Result<ReportDescriptorBuilder<BuilderWithoutReportIdxxF>> {
        Ok(ReportDescriptorBuilder {
            marker: core::marker::PhantomData,
            input_reports: None,
            output_reports: None,
            feature_reports: None,
            input_report: None,
            output_report: None,
            feature_report: Some(feature_report),
            toplevel_collections: self.toplevel_collections,
        })
    }

    /// Add an input report to this builder.
    pub fn add_input_report(
        self,
        input_report: InputReport<WithReportId>,
    ) -> Result<ReportDescriptorBuilder<BuilderWithReportId>> {
        Ok(ReportDescriptorBuilder {
            marker: core::marker::PhantomData,
            input_reports: Some(vec![input_report]),
            output_reports: Some(Vec::new()),
            feature_reports: Some(Vec::new()),
            input_report: None,
            output_report: None,
            feature_report: None,
            toplevel_collections: self.toplevel_collections,
        })
    }

    /// Add an output report to this builder.
    pub fn add_output_report(
        self,
        output_report: OutputReport<WithReportId>,
    ) -> Result<ReportDescriptorBuilder<BuilderWithReportId>> {
        Ok(ReportDescriptorBuilder {
            marker: core::marker::PhantomData,
            input_reports: Some(Vec::new()),
            output_reports: Some(vec![output_report]),
            feature_reports: Some(Vec::new()),
            input_report: None,
            output_report: None,
            feature_report: None,
            toplevel_collections: self.toplevel_collections,
        })
    }

    /// Add a feature report to this builder.
    pub fn add_feature_report(
        self,
        feature_report: FeatureReport<WithReportId>,
    ) -> Result<ReportDescriptorBuilder<BuilderWithReportId>> {
        Ok(ReportDescriptorBuilder {
            marker: core::marker::PhantomData,
            input_reports: Some(Vec::new()),
            output_reports: Some(Vec::new()),
            feature_reports: Some(vec![feature_report]),
            input_report: None,
            output_report: None,
            feature_report: None,
            toplevel_collections: self.toplevel_collections,
        })
    }
}

// BuilderWithoutReportIdIxx - only input is set
impl ReportDescriptorBuilder<BuilderWithoutReportIdIxx> {
    #[doc(hidden)]
    pub fn set_output_report(
        self,
        output_report: OutputReport<WithoutReportId>,
    ) -> Result<ReportDescriptorBuilder<BuilderWithoutReportIdIOx>> {
        Ok(ReportDescriptorBuilder {
            marker: core::marker::PhantomData,
            input_reports: self.input_reports,
            output_reports: self.output_reports,
            feature_reports: self.feature_reports,
            input_report: self.input_report,
            output_report: Some(output_report),
            feature_report: self.feature_report,
            toplevel_collections: self.toplevel_collections,
        })
    }

    #[doc(hidden)]
    pub fn set_feature_report(
        self,
        feature_report: FeatureReport<WithoutReportId>,
    ) -> Result<ReportDescriptorBuilder<BuilderWithoutReportIdIxF>> {
        Ok(ReportDescriptorBuilder {
            marker: core::marker::PhantomData,
            input_reports: self.input_reports,
            output_reports: self.output_reports,
            feature_reports: self.feature_reports,
            input_report: self.input_report,
            output_report: self.output_report,
            feature_report: Some(feature_report),
            toplevel_collections: self.toplevel_collections,
        })
    }
}

// BuilderWithoutReportIdxOx - only output is set
impl ReportDescriptorBuilder<BuilderWithoutReportIdxOx> {
    #[doc(hidden)]
    pub fn set_input_report(
        self,
        input_report: InputReport<WithoutReportId>,
    ) -> Result<ReportDescriptorBuilder<BuilderWithoutReportIdIOx>> {
        Ok(ReportDescriptorBuilder {
            marker: core::marker::PhantomData,
            input_reports: self.input_reports,
            output_reports: self.output_reports,
            feature_reports: self.feature_reports,
            input_report: Some(input_report),
            output_report: self.output_report,
            feature_report: self.feature_report,
            toplevel_collections: self.toplevel_collections,
        })
    }

    #[doc(hidden)]
    pub fn set_feature_report(
        self,
        feature_report: FeatureReport<WithoutReportId>,
    ) -> Result<ReportDescriptorBuilder<BuilderWithoutReportIdxOF>> {
        Ok(ReportDescriptorBuilder {
            marker: core::marker::PhantomData,
            input_reports: self.input_reports,
            output_reports: self.output_reports,
            feature_reports: self.feature_reports,
            input_report: self.input_report,
            output_report: self.output_report,
            feature_report: Some(feature_report),
            toplevel_collections: self.toplevel_collections,
        })
    }
}

// BuilderWithoutReportIdxxF - only feature is set
impl ReportDescriptorBuilder<BuilderWithoutReportIdxxF> {
    #[doc(hidden)]
    pub fn set_input_report(
        self,
        input_report: InputReport<WithoutReportId>,
    ) -> Result<ReportDescriptorBuilder<BuilderWithoutReportIdIxF>> {
        Ok(ReportDescriptorBuilder {
            marker: core::marker::PhantomData,
            input_reports: self.input_reports,
            output_reports: self.output_reports,
            feature_reports: self.feature_reports,
            input_report: Some(input_report),
            output_report: self.output_report,
            feature_report: self.feature_report,
            toplevel_collections: self.toplevel_collections,
        })
    }

    #[doc(hidden)]
    pub fn set_output_report(
        self,
        output_report: OutputReport<WithoutReportId>,
    ) -> Result<ReportDescriptorBuilder<BuilderWithoutReportIdxOF>> {
        Ok(ReportDescriptorBuilder {
            marker: core::marker::PhantomData,
            input_reports: self.input_reports,
            output_reports: self.output_reports,
            feature_reports: self.feature_reports,
            input_report: self.input_report,
            output_report: Some(output_report),
            feature_report: self.feature_report,
            toplevel_collections: self.toplevel_collections,
        })
    }
}

// BuilderWithoutReportIdIOx - input and output are set
impl ReportDescriptorBuilder<BuilderWithoutReportIdIOx> {
    #[doc(hidden)]
    pub fn set_feature_report(
        self,
        feature_report: FeatureReport<WithoutReportId>,
    ) -> Result<ReportDescriptorBuilder<BuilderWithoutReportIdIOF>> {
        Ok(ReportDescriptorBuilder {
            marker: core::marker::PhantomData,
            input_reports: self.input_reports,
            output_reports: self.output_reports,
            feature_reports: self.feature_reports,
            input_report: self.input_report,
            output_report: self.output_report,
            feature_report: Some(feature_report),
            toplevel_collections: self.toplevel_collections,
        })
    }
}

// BuilderWithoutReportIdIxF - input and feature are set
impl ReportDescriptorBuilder<BuilderWithoutReportIdIxF> {
    #[doc(hidden)]
    pub fn set_output_report(
        self,
        output_report: OutputReport<WithoutReportId>,
    ) -> Result<ReportDescriptorBuilder<BuilderWithoutReportIdIOF>> {
        Ok(ReportDescriptorBuilder {
            marker: core::marker::PhantomData,
            input_reports: self.input_reports,
            output_reports: self.output_reports,
            feature_reports: self.feature_reports,
            input_report: self.input_report,
            output_report: Some(output_report),
            feature_report: self.feature_report,
            toplevel_collections: self.toplevel_collections,
        })
    }
}

// BuilderWithoutReportIdxOF - output and feature are set
impl ReportDescriptorBuilder<BuilderWithoutReportIdxOF> {
    #[doc(hidden)]
    pub fn set_input_report(
        self,
        input_report: InputReport<WithoutReportId>,
    ) -> Result<ReportDescriptorBuilder<BuilderWithoutReportIdIOF>> {
        Ok(ReportDescriptorBuilder {
            marker: core::marker::PhantomData,
            input_reports: self.input_reports,
            output_reports: self.output_reports,
            feature_reports: self.feature_reports,
            input_report: Some(input_report),
            output_report: self.output_report,
            feature_report: self.feature_report,
            toplevel_collections: self.toplevel_collections,
        })
    }
}

impl<S: ReportDescriptorBuilderState> ReportDescriptorBuilder<S> {
    /// Add the given toplevel collection
    pub fn add_toplevel_collection(mut self, collection: ToplevelCollection) -> Self {
        let inner = collection.into_inner();
        self.toplevel_collections.push(inner);
        self
    }

    /// Build a HID Report Descriptor byte array.
    ///
    /// The output of this function is not guaranteed to be stable, different versions
    /// of this crate may produce different report descriptors.
    pub fn build(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();

        // Create state tracker for optimization
        let mut stack = DescriptorStack::new();

        // Emit each root collection and its tree
        for toplevel in &self.toplevel_collections {
            self.emit_collection(&mut bytes, toplevel, &mut stack)?;
        }

        Ok(bytes)
    }

    /// Emit a collection and all its contents into the byte stream
    fn emit_collection(
        &self,
        bytes: &mut Vec<u8>,
        collection: &Collection,
        stack: &mut DescriptorStack,
    ) -> Result<()> {
        use hidreport::hid::{CollectionItem, GlobalItem, ItemType, MainItem};

        // Emit usage page and usage for collection
        let usage = collection.usage();
        stack.emit_usage_page(bytes, &usage.usage_page);
        stack.emit_usage_id(bytes, &usage.usage_id);

        // Emit collection open
        let collection_type = match collection {
            Collection::Application(_) => CollectionItem::Application,
            Collection::Physical(_) => CollectionItem::Physical,
            Collection::Logical(_) => CollectionItem::Logical,
        };
        bytes.extend_from_slice(&ItemType::Main(MainItem::Collection(collection_type)).as_bytes());

        // // Push stack for this collection's scope
        // bytes.extend_from_slice(&ItemType::Global(GlobalItem::Push).as_bytes());

        // Group all fields by report ID
        let collection_id = collection.id();
        let fields_by_report = self.group_fields_by_report(&collection_id);

        // Process each report ID
        for (report_id, (input_fields, output_fields, feature_fields)) in fields_by_report.iter() {
            // Emit report ID if present
            if let Some(rid) = report_id {
                bytes.extend_from_slice(&ItemType::Global(GlobalItem::ReportId(*rid)).as_bytes());
            }

            // Emit fields in order input, output, feature.
            // This is merely the simplest strategy, it may be that a input and feature share the
            // same stack while the output field is different but that's niche enough so let's
            // leave that for the future.
            for field in input_fields {
                self.emit_field(bytes, field, stack, ReportType::Input)?;
            }

            for field in output_fields {
                self.emit_field(bytes, field, stack, ReportType::Output)?;
            }

            for field in feature_fields {
                self.emit_field(bytes, field, stack, ReportType::Feature)?;
            }
        }

        // Recursively emit child collections
        for child in collection.children() {
            self.emit_collection(bytes, child, stack)?;
        }

        // // Pop stack
        // bytes.extend_from_slice(&ItemType::Global(GlobalItem::Pop).as_bytes());

        // Emit collection close
        bytes.extend_from_slice(&ItemType::Main(MainItem::EndCollection).as_bytes());

        Ok(())
    }

    /// Group fields by report ID and type for a given collection,
    /// tuple items are input, output and feature report fields.
    ///
    /// This isn't necessary the best strategy but it's probably the most common one.
    fn group_fields_by_report(
        &self,
        collection_id: &CollectionId,
    ) -> BTreeMap<Option<ReportId>, (Vec<&Field>, Vec<&Field>, Vec<&Field>)> {
        let mut grouped: BTreeMap<Option<ReportId>, (Vec<&Field>, Vec<&Field>, Vec<&Field>)> =
            BTreeMap::new();

        // Handle reports with ReportId
        if let Some(input_reports) = &self.input_reports {
            for report in input_reports {
                for field in &report.fields {
                    if let Some(field_collection_id) = field.collection() {
                        if field_collection_id == collection_id {
                            let entry = grouped.entry(report.report_id).or_insert((
                                Vec::new(),
                                Vec::new(),
                                Vec::new(),
                            ));
                            entry.0.push(field);
                        }
                    }
                }
            }
        }

        if let Some(output_reports) = &self.output_reports {
            for report in output_reports {
                for field in &report.fields {
                    if let Some(field_collection_id) = field.collection() {
                        if field_collection_id == collection_id {
                            let entry = grouped.entry(report.report_id).or_insert((
                                Vec::new(),
                                Vec::new(),
                                Vec::new(),
                            ));
                            entry.1.push(field);
                        }
                    }
                }
            }
        }

        if let Some(feature_reports) = &self.feature_reports {
            for report in feature_reports {
                for field in &report.fields {
                    if let Some(field_collection_id) = field.collection() {
                        if field_collection_id == collection_id {
                            let entry = grouped.entry(report.report_id).or_insert((
                                Vec::new(),
                                Vec::new(),
                                Vec::new(),
                            ));
                            entry.2.push(field);
                        }
                    }
                }
            }
        }

        // Handle reports without ReportId
        if let Some(input_report) = &self.input_report {
            for field in &input_report.fields {
                if let Some(field_collection_id) = field.collection() {
                    if field_collection_id == collection_id {
                        let entry = grouped.entry(input_report.report_id).or_insert((
                            Vec::new(),
                            Vec::new(),
                            Vec::new(),
                        ));
                        entry.0.push(field);
                    }
                }
            }
        }

        if let Some(output_report) = &self.output_report {
            for field in &output_report.fields {
                if let Some(field_collection_id) = field.collection() {
                    if field_collection_id == collection_id {
                        let entry = grouped.entry(output_report.report_id).or_insert((
                            Vec::new(),
                            Vec::new(),
                            Vec::new(),
                        ));
                        entry.1.push(field);
                    }
                }
            }
        }

        if let Some(feature_report) = &self.feature_report {
            for field in &feature_report.fields {
                if let Some(field_collection_id) = field.collection() {
                    if field_collection_id == collection_id {
                        let entry = grouped.entry(feature_report.report_id).or_insert((
                            Vec::new(),
                            Vec::new(),
                            Vec::new(),
                        ));
                        entry.2.push(field);
                    }
                }
            }
        }

        grouped
    }

    /// Emit a single field to the byte stream
    fn emit_field(
        &self,
        bytes: &mut Vec<u8>,
        field: &Field,
        stack: &mut DescriptorStack,
        report_type: ReportType,
    ) -> Result<()> {
        use hidreport::hid::{GlobalItem, ItemBuilder, ItemType, LocalItem, MainItem};

        match field {
            Field::Variable(var) => {
                // Emit global attributes
                stack.emit_logical_range(bytes, var.logical_minimum, var.logical_maximum);
                stack.emit_physical_range(bytes, var.physical_minimum, var.physical_maximum);
                stack.emit_unit(bytes, var.unit);
                stack.emit_unit_exponent(bytes, var.exponent);

                // Emit report size and count
                bytes.extend_from_slice(
                    &ItemType::Global(GlobalItem::ReportSize(ReportSize::from(var.size_in_bits)))
                        .as_bytes(),
                );
                bytes.extend_from_slice(
                    &ItemType::Global(GlobalItem::ReportCount(ReportCount::from(1))).as_bytes(),
                );

                // Emit usage
                stack.emit_usage_page(bytes, &var.usage.usage_page);
                bytes.extend_from_slice(
                    &ItemType::Local(LocalItem::UsageId(var.usage.usage_id)).as_bytes(),
                );

                // Emit main item (variable, absolute, data)
                let item = ItemBuilder::new().data().variable().absolute();
                let main_item = match report_type {
                    ReportType::Input => MainItem::Input(item.input()),
                    ReportType::Output => MainItem::Output(item.output()),
                    ReportType::Feature => MainItem::Feature(item.feature()),
                };
                bytes.extend_from_slice(&ItemType::Main(main_item).as_bytes());
            }
            Field::Array(arr) => {
                // Emit global attributes
                stack.emit_logical_range(bytes, arr.logical_minimum, arr.logical_maximum);
                stack.emit_physical_range(bytes, arr.physical_minimum, arr.physical_maximum);
                stack.emit_unit(bytes, arr.unit);
                stack.emit_unit_exponent(bytes, arr.exponent);

                // Emit report size and count
                bytes.extend_from_slice(
                    &ItemType::Global(GlobalItem::ReportSize(ReportSize::from(arr.size_in_bits)))
                        .as_bytes(),
                );
                bytes.extend_from_slice(
                    &ItemType::Global(GlobalItem::ReportCount(ReportCount::from(arr.count)))
                        .as_bytes(),
                );

                // Emit usages or usage range
                if let Some((min, max)) = arr.usage_range {
                    // Emit usage page before usage minimum/maximum
                    let usage_page = min.usage_page();
                    stack.emit_usage_page(bytes, &usage_page);

                    bytes.extend_from_slice(
                        &ItemType::Local(LocalItem::UsageMinimum(min)).as_bytes(),
                    );
                    bytes.extend_from_slice(
                        &ItemType::Local(LocalItem::UsageMaximum(max)).as_bytes(),
                    );
                } else {
                    for usage in &arr.usages {
                        stack.emit_usage_page(bytes, &usage.usage_page);
                        bytes.extend_from_slice(
                            &ItemType::Local(LocalItem::UsageId(usage.usage_id)).as_bytes(),
                        );
                    }
                }

                // Emit main item (array, absolute, data)
                let item = ItemBuilder::new().data().array().absolute();
                let main_item = match report_type {
                    ReportType::Input => MainItem::Input(item.input()),
                    ReportType::Output => MainItem::Output(item.output()),
                    ReportType::Feature => MainItem::Feature(item.feature()),
                };
                bytes.extend_from_slice(&ItemType::Main(main_item).as_bytes());
            }
            Field::Constant(constant) => {
                // Emit report size and count
                bytes.extend_from_slice(
                    &ItemType::Global(GlobalItem::ReportSize(ReportSize::from(
                        constant.size_in_bits,
                    )))
                    .as_bytes(),
                );
                bytes.extend_from_slice(
                    &ItemType::Global(GlobalItem::ReportCount(ReportCount::from(1))).as_bytes(),
                );

                // Emit constant main item
                let item = ItemBuilder::new().constant();
                let main_item = match report_type {
                    ReportType::Input => MainItem::Input(item.input()),
                    ReportType::Output => MainItem::Output(item.output()),
                    ReportType::Feature => MainItem::Feature(item.feature()),
                };
                bytes.extend_from_slice(&ItemType::Main(main_item).as_bytes());
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collection_factory_methods() {
        let usage = Usage::from_page_and_id(0x01.into(), 0x02.into());

        let app = Collection::application(usage);
        assert!(matches!(app, Collection::Application(_)));

        let phys = Collection::physical(usage);
        assert!(matches!(phys, Collection::Physical(_)));

        let logical = Collection::logical(usage);
        assert!(matches!(logical, Collection::Logical(_)));
    }

    #[test]
    fn test_collection_add_to() {
        let usages: Vec<Usage> = (1..=3)
            .map(|i| Usage::from_page_and_id(0x01.into(), i.into()))
            .collect();

        let mut app = Collection::application(usages[0]);
        let mut phys = Collection::physical(usages[1]);
        let logical = Collection::logical(usages[2]);

        let logical_id = logical.add_to(&mut phys);
        let phys_id = phys.add_to(&mut app);

        assert_eq!(app.children().len(), 1);
        assert_eq!(app.children()[0].children().len(), 1);
        assert_eq!(app.children()[0].id(), phys_id);
        assert_eq!(app.children()[0].children()[0].id(), logical_id);
    }

    #[test]
    fn test_field_builder() {
        let usages: Vec<Usage> = (1..=3)
            .map(|i| Usage::from_page_and_id(0x01.into(), i.into()))
            .collect();
        let mut app = Collection::application(usages[0]);
        let phys = Collection::physical(usages[1]);
        let coll_id = phys.add_to(&mut app);

        let _field = FieldBuilder::new(8.try_into().unwrap())
            .logical_range(&LogicalMinimum::from(-128), &LogicalMaximum::from(127))
            .collection(&coll_id)
            .unit(Unit::from(0x01))
            .unit_exponent(UnitExponent::from(0))
            .variable()
            .usage(&usages[2])
            .build();

        let _field = FieldBuilder::new(1.try_into().unwrap())
            .logical_range(&LogicalMinimum::from(0), &LogicalMaximum::from(1))
            .collection(&coll_id)
            .unit(Unit::from(0x01))
            .unit_exponent(UnitExponent::from(0))
            .array()
            .usage_range(&UsageMinimum::from(1), &UsageMaximum::from(10))
            .count(10.try_into().unwrap())
            .unwrap()
            .build();

        let _field = FieldBuilder::new(6.try_into().unwrap())
            .collection(&coll_id)
            .constant()
            .build();
        // without collection
        let _field = FieldBuilder::new(6.try_into().unwrap()).constant().build();
    }

    /// Test ReportBuilder creating input reports
    #[test]
    fn test_report_builder_input() {
        let usages: Vec<Usage> = (1..=3)
            .map(|i| Usage::from_page_and_id(0x01.into(), i.into()))
            .collect();
        let mut app = Collection::application(usages[0]);
        let phys = Collection::physical(usages[1]);
        let coll_id = phys.add_to(&mut app);

        let field = FieldBuilder::new(8.try_into().unwrap())
            .collection(&coll_id)
            .variable()
            .usage(&usages[2])
            .build();

        let _report = ReportBuilder::new()
            .report_id(ReportId::from(1))
            .append_field(field)
            .input_report();

        // Report is successfully built
    }

    /// Test ReportBuilder with padding
    #[test]
    fn test_report_builder_padding() {
        let usages: Vec<Usage> = (1..=3)
            .map(|i| Usage::from_page_and_id(0x01.into(), i.into()))
            .collect();
        let mut app = Collection::application(usages[0]);
        let phys = Collection::physical(usages[1]);
        let coll_id = phys.add_to(&mut app);

        let field = FieldBuilder::new(5.try_into().unwrap())
            .collection(&coll_id)
            .variable()
            .usage(&usages[2])
            .build();

        let _report = ReportBuilder::new()
            .report_id(ReportId::from(1))
            .append_field(field)
            .pad_to_8()
            .input_report();

        // Report is successfully built with padding
    }

    /// Test ReportBuilder pad_to_16
    #[test]
    fn test_report_builder_pad_to_16() {
        let usages: Vec<Usage> = (1..=3)
            .map(|i| Usage::from_page_and_id(0x01.into(), i.into()))
            .collect();
        let mut app = Collection::application(usages[0]);
        let phys = Collection::physical(usages[1]);
        let coll_id = phys.add_to(&mut app);

        let field = FieldBuilder::new(10.try_into().unwrap())
            .collection(&coll_id)
            .variable()
            .usage(&usages[2])
            .build();

        let _report = ReportBuilder::new()
            .append_field(field)
            .pad_to_16()
            .input_report();

        // Report is successfully built with padding
    }

    /// Test ReportDescriptorBuilder adding collections
    #[test]
    fn test_descriptor_builder_add_collection() {
        let usage = Usage::from_page_and_id(UsagePage::from(0x01), UsageId::from(0x02));
        let app = Collection::application(usage);

        let _builder = ReportDescriptorBuilder::new()
            .add_toplevel_collection(app.toplevel());

        // Successfully added collection
    }

    /// Test ReportDescriptorBuilder adding input reports
    #[test]
    fn test_descriptor_builder_add_input_report() {
        let usage = Usage::from_page_and_id(UsagePage::from(0x01), UsageId::from(0x30));
        let mut app = Collection::application(Usage::from_page_and_id(
            UsagePage::from(0x01),
            UsageId::from(0x02),
        ));
        let phys = Collection::physical(Usage::from_page_and_id(
            UsagePage::from(0x01),
            UsageId::from(0x03),
        ));
        let coll_id = phys.add_to(&mut app);

        let field = FieldBuilder::new(8.try_into().unwrap())
            .collection(&coll_id)
            .variable()
            .usage(&usage)
            .build();

        let report = ReportBuilder::new()
            .report_id(ReportId::from(1))
            .append_field(field)
            .input_report();

        let _builder = ReportDescriptorBuilder::new()
            .add_input_report(report)
            .expect("Should add input report");

        // Successfully added input report
    }

    /// Test ReportDescriptorBuilder duplicate report ID detection
    #[test]
    fn test_descriptor_builder_duplicate_report_id() {
        let usage = Usage::from_page_and_id(UsagePage::from(0x01), UsageId::from(0x30));
        let mut app = Collection::application(Usage::from_page_and_id(
            UsagePage::from(0x01),
            UsageId::from(0x02),
        ));
        let phys = Collection::physical(Usage::from_page_and_id(
            UsagePage::from(0x01),
            UsageId::from(0x03),
        ));
        let coll_id = phys.add_to(&mut app);

        let field1 = FieldBuilder::new(8.try_into().unwrap())
            .collection(&coll_id)
            .variable()
            .usage(&usage)
            .build();

        let field2 = FieldBuilder::new(8.try_into().unwrap())
            .collection(&coll_id)
            .variable()
            .usage(&usage)
            .build();

        let report1 = ReportBuilder::new()
            .report_id(ReportId::from(1))
            .append_field(field1)
            .input_report();

        let report2 = ReportBuilder::new()
            .report_id(ReportId::from(1))
            .append_field(field2)
            .input_report();

        let result = ReportDescriptorBuilder::new()
            .add_input_report(report1)
            .and_then(|b| b.add_input_report(report2));

        assert!(result.is_err(), "Should reject duplicate report IDs");
    }

    /// Test building a complete HID descriptor
    #[test]
    fn test_build_complete_descriptor() {
        let usage_mouse = Usage::from_page_and_id(UsagePage::from(0x01), UsageId::from(0x02));
        let usage_pointer = Usage::from_page_and_id(UsagePage::from(0x01), UsageId::from(0x01));
        let usage_x = Usage::from_page_and_id(UsagePage::from(0x01), UsageId::from(0x30));
        let usage_y = Usage::from_page_and_id(UsagePage::from(0x01), UsageId::from(0x31));

        let mut app = Collection::application(usage_mouse);
        let phys = Collection::physical(usage_pointer);
        let coll_id = phys.add_to(&mut app);

        let x = FieldBuilder::new(8.try_into().unwrap())
            .logical_range(&LogicalMinimum::from(-128), &LogicalMaximum::from(127))
            .collection(&coll_id)
            .variable()
            .usage(&usage_x)
            .build();

        let y = FieldBuilder::new(8.try_into().unwrap())
            .logical_range(&LogicalMinimum::from(-128), &LogicalMaximum::from(127))
            .collection(&coll_id)
            .variable()
            .usage(&usage_y)
            .build();

        let report = ReportBuilder::new()
            .report_id(ReportId::from(1))
            .append_field(x)
            .append_field(y)
            .input_report();

        let bytes = ReportDescriptorBuilder::new()
            .add_toplevel_collection(app.toplevel())
            .add_input_report(report)
            .expect("Should add report")
            .build()
            .expect("Should build descriptor");

        assert!(!bytes.is_empty(), "Descriptor should contain bytes");

        // Verify it starts with Usage Page and Usage (typical HID descriptor structure)
        // Usage Page: 0x05 prefix + 1 byte data
        // Usage: 0x09 prefix + 1 byte data
        assert_eq!(bytes[0] & 0xFC, 0x04, "Should start with Usage Page");
    }

    /// Test building descriptor with multiple reports
    #[test]
    fn test_build_descriptor_multiple_reports() {
        let usage_mouse = Usage::from_page_and_id(UsagePage::from(0x01), UsageId::from(0x02));
        let usage_pointer = Usage::from_page_and_id(UsagePage::from(0x01), UsageId::from(0x01));
        let usage_x = Usage::from_page_and_id(UsagePage::from(0x01), UsageId::from(0x30));

        let mut app = Collection::application(usage_mouse);
        let phys = Collection::physical(usage_pointer);
        let coll_id = phys.add_to(&mut app);

        let field1 = FieldBuilder::new(8.try_into().unwrap())
            .collection(&coll_id)
            .variable()
            .usage(&usage_x)
            .build();

        let field2 = FieldBuilder::new(8.try_into().unwrap())
            .collection(&coll_id)
            .variable()
            .usage(&usage_x)
            .build();

        let input_report = ReportBuilder::new()
            .report_id(ReportId::from(1))
            .append_field(field1)
            .input_report();

        let output_report = ReportBuilder::new()
            .report_id(ReportId::from(2))
            .append_field(field2)
            .output_report();

        let bytes = ReportDescriptorBuilder::new()
            .add_toplevel_collection(app.toplevel())
            .add_input_report(input_report)
            .expect("Should add input report")
            .add_output_report(output_report)
            .expect("Should add output report")
            .build()
            .expect("Should build descriptor");

        assert!(!bytes.is_empty(), "Descriptor should contain bytes");
    }

    /// Test output and feature reports
    #[test]
    fn test_output_and_feature_reports() {
        let usage = Usage::from_page_and_id(UsagePage::from(0x01), UsageId::from(0x30));
        let mut app = Collection::application(Usage::from_page_and_id(
            UsagePage::from(0x01),
            UsageId::from(0x02),
        ));
        let phys = Collection::physical(Usage::from_page_and_id(
            UsagePage::from(0x01),
            UsageId::from(0x03),
        ));
        let coll_id = phys.add_to(&mut app);

        let field1 = FieldBuilder::new(8.try_into().unwrap())
            .collection(&coll_id)
            .variable()
            .usage(&usage)
            .build();

        let field2 = FieldBuilder::new(8.try_into().unwrap())
            .collection(&coll_id)
            .variable()
            .usage(&usage)
            .build();

        let _output = ReportBuilder::new()
            .report_id(ReportId::from(1))
            .append_field(field1)
            .output_report();

        let _feature = ReportBuilder::new()
            .report_id(ReportId::from(2))
            .append_field(field2)
            .feature_report();

        // Reports successfully created
    }

    /// Test with actual HUT usages
    #[test]
    fn test_with_hut_usages() {
        use hut::{AsUsage, GenericDesktop};

        let mut app = Collection::application(GenericDesktop::Mouse.usage().into());
        let phys = Collection::physical(GenericDesktop::Pointer.usage().into());
        let coll_id = phys.add_to(&mut app);

        let usage_x = GenericDesktop::X.usage().into();
        let usage_y = GenericDesktop::Y.usage().into();

        let x = FieldBuilder::new(8.try_into().unwrap())
            .logical_range(&LogicalMinimum::from(-128), &LogicalMaximum::from(127))
            .collection(&coll_id)
            .variable()
            .usage(&usage_x)
            .build();

        let y = FieldBuilder::new(8.try_into().unwrap())
            .logical_range(&LogicalMinimum::from(-128), &LogicalMaximum::from(127))
            .collection(&coll_id)
            .variable()
            .usage(&usage_y)
            .build();

        let report = ReportBuilder::new()
            .report_id(ReportId::from(1))
            .append_field(x)
            .append_field(y)
            .input_report();

        let bytes = ReportDescriptorBuilder::new()
            .add_toplevel_collection(app.toplevel())
            .add_input_report(report)
            .expect("Should add report")
            .build()
            .expect("Should build descriptor");

        assert!(!bytes.is_empty());
    }

    /// Test round-trip: build descriptor and parse it back
    #[test]
    fn test_builder_round_trip() {
        use hidreport::ReportDescriptor;

        let usage_mouse = Usage::from_page_and_id(UsagePage::from(0x01), UsageId::from(0x02));
        let usage_pointer = Usage::from_page_and_id(UsagePage::from(0x01), UsageId::from(0x01));
        let usage_x = Usage::from_page_and_id(UsagePage::from(0x01), UsageId::from(0x30));
        let usage_y = Usage::from_page_and_id(UsagePage::from(0x01), UsageId::from(0x31));

        let mut app = Collection::application(usage_mouse);
        let phys = Collection::physical(usage_pointer);
        let coll_id = phys.add_to(&mut app);

        let x = FieldBuilder::new(8.try_into().unwrap())
            .logical_range(&LogicalMinimum::from(-128), &LogicalMaximum::from(127))
            .collection(&coll_id)
            .variable()
            .usage(&usage_x)
            .build();

        let y = FieldBuilder::new(8.try_into().unwrap())
            .logical_range(&LogicalMinimum::from(-128), &LogicalMaximum::from(127))
            .collection(&coll_id)
            .variable()
            .usage(&usage_y)
            .build();

        let report = ReportBuilder::new()
            .report_id(ReportId::from(1))
            .append_field(x)
            .append_field(y)
            .input_report();

        let bytes = ReportDescriptorBuilder::new()
            .add_toplevel_collection(app.toplevel())
            .add_input_report(report)
            .expect("Should add report")
            .build()
            .expect("Should build descriptor");

        // Parse the descriptor back
        let rdesc =
            ReportDescriptor::try_from(bytes.as_slice()).expect("Should parse built descriptor");

        // Verify the parsed descriptor has the expected structure
        let input_reports = rdesc.input_reports();
        assert_eq!(
            input_reports.len(),
            1,
            "Should have exactly one input report"
        );

        let report = &input_reports[0];
        assert_eq!(
            report.report_id(),
            &Some(ReportId::from(1)),
            "Report should have ID 1"
        );

        // Should have 2 variable fields (x and y)
        let variable_fields: Vec<_> = report
            .fields()
            .iter()
            .filter_map(|f| match f {
                hidreport::Field::Variable(v) => Some(v),
                _ => None,
            })
            .collect();

        assert_eq!(variable_fields.len(), 2, "Should have 2 variable fields");

        // Verify x field
        assert_eq!(
            u32::from(&variable_fields[0].usage),
            0x00010030,
            "First field should be X usage"
        );
        assert_eq!(
            variable_fields[0].logical_minimum,
            LogicalMinimum::from(-128)
        );
        assert_eq!(
            variable_fields[0].logical_maximum,
            LogicalMaximum::from(127)
        );

        // Verify y field
        assert_eq!(
            u32::from(&variable_fields[1].usage),
            0x00010031,
            "Second field should be Y usage"
        );
        assert_eq!(
            variable_fields[1].logical_minimum,
            LogicalMinimum::from(-128)
        );
        assert_eq!(
            variable_fields[1].logical_maximum,
            LogicalMaximum::from(127)
        );
    }

    /// Test round-trip with array field and usage range
    #[test]
    fn test_builder_round_trip_array() {
        use hidreport::ReportDescriptor;

        let usage_mouse = Usage::from_page_and_id(UsagePage::from(0x01), UsageId::from(0x02));
        let usage_pointer = Usage::from_page_and_id(UsagePage::from(0x01), UsageId::from(0x01));

        let mut app = Collection::application(usage_mouse);
        let phys = Collection::physical(usage_pointer);
        let coll_id = phys.add_to(&mut app);

        // Create an array field with button usages
        let buttons = FieldBuilder::new(1.try_into().unwrap())
            .logical_range(&LogicalMinimum::from(0), &LogicalMaximum::from(1))
            .collection(&coll_id)
            .array()
            .usage_range(
                &UsageMinimum::from(0x00090001),
                &UsageMaximum::from(0x00090008),
            )
            .count(8.try_into().unwrap())
            .unwrap()
            .build();

        let report = ReportBuilder::new()
            .report_id(ReportId::from(1))
            .append_field(buttons)
            .input_report();

        let bytes = ReportDescriptorBuilder::new()
            .add_toplevel_collection(app.toplevel())
            .add_input_report(report)
            .expect("Should add report")
            .build()
            .expect("Should build descriptor");

        // Parse the descriptor back
        let rdesc =
            ReportDescriptor::try_from(bytes.as_slice()).expect("Should parse built descriptor");

        // Verify the parsed descriptor
        let input_reports = rdesc.input_reports();
        assert_eq!(
            input_reports.len(),
            1,
            "Should have exactly one input report"
        );

        let report = &input_reports[0];
        assert_eq!(
            report.report_id(),
            &Some(ReportId::from(1)),
            "Report should have ID 1"
        );

        // Should have 1 array field
        let array_fields: Vec<_> = report
            .fields()
            .iter()
            .filter_map(|f| match f {
                hidreport::Field::Array(a) => Some(a),
                _ => None,
            })
            .collect();

        assert_eq!(array_fields.len(), 1, "Should have 1 array field");

        let array = array_fields[0];
        assert_eq!(
            usize::from(array.report_count),
            8,
            "Array should have count of 8"
        );

        // Verify usage range
        let usage_range = array.usage_range().expect("Array should have usage range");
        assert_eq!(
            usage_range.minimum(),
            UsageMinimum::from(0x00090001),
            "Usage minimum should be button 1"
        );
        assert_eq!(
            usage_range.maximum(),
            UsageMaximum::from(0x00090008),
            "Usage maximum should be button 8"
        );
    }

    /// Test that count() validates size_in_bits is large enough for the logical range
    #[test]
    fn test_count_validates_size() {
        let usage = Usage::from_page_and_id(UsagePage::from(0x01), UsageId::from(0x30));
        let mut app = Collection::application(usage);
        let phys = Collection::physical(usage);
        let coll_id = phys.add_to(&mut app);

        // This should succeed: 4 bits is enough for range 0-10 (needs 4 bits)
        let _field = FieldBuilder::new(4.try_into().unwrap())
            .logical_range(&LogicalMinimum::from(0), &LogicalMaximum::from(10))
            .collection(&coll_id)
            .array()
            .usage_range(&UsageMinimum::from(1), &UsageMaximum::from(10))
            .count(5.try_into().unwrap())
            .expect("Should succeed with 4 bits for range 0-10")
            .build();

        // This should fail: 2 bits is not enough for range 0-10 (needs 4 bits)
        let result = FieldBuilder::new(2.try_into().unwrap())
            .logical_range(&LogicalMinimum::from(0), &LogicalMaximum::from(10))
            .collection(&coll_id)
            .array()
            .usage_range(&UsageMinimum::from(1), &UsageMaximum::from(10))
            .count(5.try_into().unwrap());

        assert!(result.is_err(), "Should fail with 2 bits for range 0-10");
        if let Err(err) = result {
            assert!(
                matches!(err, BuilderError::Size),
                "Should return Size error"
            );
        }
    }
}
