use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Write},
    mem::MaybeUninit,
    ops::{Deref, Index},
    path::Path,
};

pub(crate) use akari_common::{base64, memmap2, rayon, serde, serde_json};
pub(crate) use akari_cpp_ext as cpp_ext;
use serde::{de::*, ser::*};
pub(crate) use serde::{Deserialize, Serialize};
pub mod blender;
pub mod scene;
pub mod shader;
pub mod parser;

pub use scene::*;
pub use shader::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(crate = "serde")]
pub struct NodeRef<T> {
    pub id: String,
    #[serde(skip)]
    phantom: std::marker::PhantomData<T>,
}
impl<T> std::fmt::Display for NodeRef<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.id)
    }
}
impl<T> NodeRef<T> {
    pub fn new(id: String) -> Self {
        Self {
            id,
            phantom: std::marker::PhantomData,
        }
    }
}
impl<T> PartialEq for NodeRef<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl<T> Eq for NodeRef<T> {}
impl<T> PartialOrd for NodeRef<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.id.cmp(&other.id))
    }
}
impl<T> Ord for NodeRef<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}
impl<T> std::hash::Hash for NodeRef<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state)
    }
}
impl<T> Deref for NodeRef<T> {
    type Target = String;
    fn deref(&self) -> &Self::Target {
        &self.id
    }
}

#[derive(Clone, Debug)]
pub struct Collection<T>(BTreeMap<NodeRef<T>, T>);
impl<'a, T> IntoIterator for &'a Collection<T> {
    type Item = (&'a NodeRef<T>, &'a T);
    type IntoIter = std::collections::btree_map::Iter<'a, NodeRef<T>, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}
impl<T> std::ops::Deref for Collection<T> {
    type Target = BTreeMap<NodeRef<T>, T>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<T> std::ops::DerefMut for Collection<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
impl<T> Serialize for Collection<T>
where
    T: Serialize,
{
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.len()))?;
        for (key, value) in self.iter() {
            map.serialize_entry(&key.id, value)?;
        }
        map.end()
    }
}
impl<T> Default for Collection<T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T> Collection<T> {
    pub fn new() -> Self {
        Self(BTreeMap::new())
    }
    /// creates a new node reference that can be pushed into the collection
    pub fn new_ref(&self, hint: Option<String>) -> NodeRef<T> {
        let node_name = hint.unwrap_or_else(|| "node".to_string());
        let mut i = 0;
        let mut node_name = NodeRef {
            id: node_name,
            phantom: std::marker::PhantomData,
        };
        while self.contains_key(&node_name) {
            node_name.id = format!("{}_{}", node_name.id, i);
            i += 1;
        }
        node_name
    }
    pub fn insert(&mut self, key: NodeRef<T>, value: T) {
        if self.contains_key(&key) {
            panic!("key already exists");
        }
        self.0.insert(key, value);
    }
    pub fn update(&mut self, key: NodeRef<T>, value: T) {
        if !self.contains_key(&key) {
            panic!("key does not exist");
        }
        self.0.insert(key, value);
    }
    pub fn inner(&self) -> &BTreeMap<NodeRef<T>, T> {
        &self.0
    }
    pub fn inner_mut(&mut self) -> &mut BTreeMap<NodeRef<T>, T> {
        &mut self.0
    }
}
struct CollectionVisitor<T> {
    phantom: std::marker::PhantomData<T>,
}
impl<'de, T: Deserialize<'de>> Visitor<'de> for CollectionVisitor<T> {
    type Value = Collection<T>;
    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a map with string keys and values of type T")
    }
    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        let mut collection = Collection(BTreeMap::new());
        while let Some((key, value)) = map.next_entry::<String, T>()? {
            collection.insert(
                NodeRef {
                    id: key,
                    phantom: std::marker::PhantomData,
                },
                value,
            );
        }
        Ok(collection)
    }
}
impl<'de, T: Deserialize<'de>> Deserialize<'de> for Collection<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_map(CollectionVisitor {
            phantom: std::marker::PhantomData,
        })
    }
}

/// A slice that is serialized as a pointer and length
/// This is useful for passing slices to C libraries/Python
/// However, saving the serialized data to a file will not work
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(crate = "serde")]
pub struct ExtSlice {
    ptr: u64,
    len: u64,
}
impl ExtSlice {
    #[inline]
    pub fn new(ptr: u64, len: u64) -> Self {
        Self { ptr, len }
    }
    #[inline]
    pub fn as_ptr<T>(&self) -> *const T {
        self.ptr as *const T
    }
    #[inline]
    pub fn as_mut_ptr<T>(&self) -> *mut T {
        self.ptr as *mut T
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    #[inline]
    pub unsafe fn as_slice(&self) -> &[u8] {
        std::slice::from_raw_parts(self.ptr as *const u8, self.len as usize)
    }
    #[inline]
    pub unsafe fn as_mut_slice(&self) -> &mut [u8] {
        std::slice::from_raw_parts_mut(self.ptr as *mut u8, self.len as usize)
    }
}

/// Python script that **exports** blender scenes to akari scenes
pub const BLENDER_EXPORTER_SRC: &str = include_str!("../python/exporter.py");
