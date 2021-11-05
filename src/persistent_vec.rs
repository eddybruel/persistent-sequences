use {
    crate::{btree, btree::BTree},
    std::{
        iter::FromIterator,
        ops::{Add, Index, RangeBounds},
    },
};

/// A persistent vector.
#[derive(Clone, Debug)]
pub struct PersistentVec<T> {
    tree: BTree<T, ()>,
}

impl<T> PersistentVec<T> {
    /// Creates a new empty [`PersistentVec`].
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentVec;
    ///
    /// let vec: PersistentVec<i32> = PersistentVec::new();
    /// ```
    pub fn new() -> Self {
        Self { tree: BTree::new() }
    }

    /// Return `true` if this [`PersistentVec`] is empty.
    ///
    /// This method takes O(1) time.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentVec;
    ///
    /// let vec = PersistentVec::new();
    /// assert!(vec.is_empty());
    /// let vec = vec.push_back(0);
    /// assert!(!vec.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.tree.is_empty()
    }

    /// Returns the length of this [`PersistentVec`].
    ///
    /// This method takes O(1) time.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentVec;
    ///
    /// let vec = PersistentVec::new();
    /// assert_eq!(vec.len(), 0);
    /// let vec = vec.push_back(0);
    /// assert_eq!(vec.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.tree.len()
    }

    /// Returns a reference to the item at the front of this [`PersistentVec`], or [`None`] if the
    /// [`PersistentVec`] is empty.
    ///
    /// This method takes O(log(n)) time, where `n` is the length of the [`PersistentVec`].
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentVec;
    ///
    /// let vec = PersistentVec::new();
    /// assert_eq!(vec.front(), None);
    /// let vec = vec.push_front(0);
    /// let vec = vec.push_front(1);
    /// assert_eq!(vec.front(), Some(&1));
    /// ```
    pub fn front(&self) -> Option<&'_ T> {
        self.tree.front()
    }

    /// Returns a reference to the item at the back of this [`PersistentVec`], or [`None`] if the
    /// [`PersistentVec`] is empty.
    ///
    /// This method takes O(log(n)) time, where `n` is the length of the [`PersistentVec`].
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentVec;
    ///
    /// let vec = PersistentVec::new();
    /// assert_eq!(vec.back(), None);
    /// let vec = vec.push_back(0);
    /// let vec = vec.push_back(1);
    /// assert_eq!(vec.back(), Some(&1));
    /// ```
    pub fn back(&self) -> Option<&'_ T> {
        self.tree.back()
    }

    /// Returns a reference to the item at the given index in this [`PersistentVec`], or [`None`] if
    /// `index >= self.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentVec;
    ///
    /// let mut vec = PersistentVec::from(vec![0, 1, 2]);
    /// assert_eq!(vec.get(1), Some(&1));
    /// ```
    pub fn get(&self, index: usize) -> Option<&'_ T> {
        self.tree.get(index)
    }

    /// Returns an iterator over the items of this [`PersistentVec`].
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentVec;
    ///
    /// let vec = PersistentVec::from(vec![0, 1]);
    /// let mut iter = vec.iter();
    /// assert_eq!(iter.next(), Some(&0));
    /// assert_eq!(iter.next(), Some(&1));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            iter: self.tree.iter(),
        }
    }

    /// Returns an iterator over the given range of items in this [`PersistentVec`].
    ///
    /// # Panics
    ///
    /// Panics if the start of the range is greater than the end of the range, or the end of the
    /// range is greater than the length of the [`PersistentVec`].
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentVec;
    ///
    /// let vec = PersistentVec::from(vec![0, 1, 2, 3]);
    /// let mut range = vec.range(1..3);
    /// assert_eq!(range.next(), Some(&1));
    /// assert_eq!(range.next(), Some(&2));
    /// assert_eq!(range.next(), None);
    /// ```
    pub fn range<R: RangeBounds<usize>>(&self, range: R) -> Range<'_, T> {
        Range {
            range: self.tree.range(range),
        }
    }
}

impl<T: Clone> PersistentVec<T> {
    /// Replaces the given range of items in this [`PersistentVec`] with the given
    /// [`PersistentVec`].
    ///
    /// This method takes O(log n) time, where n is the sum of the lengths of the two
    /// [`PersistentVec`]s.
    ///
    /// # Panics
    ///
    /// Panics if the start of the range is greater than the end of the range, or the end of the
    /// range is greater than the length of the [`PersistentVec`].
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentVec;
    ///
    /// let vec_0 = PersistentVec::from(vec![0, 1, 2, 3]);
    /// let vec_1 = PersistentVec::from(vec![4, 5]);
    /// let vec = vec_0.splice(1..3, vec_1);
    /// assert_eq!(vec, PersistentVec::from(vec![0, 4, 5, 3]));
    /// ```
    pub fn splice<R: RangeBounds<usize>>(self, range: R, vec: Self) -> Self {
        Self {
            tree: self.tree.splice(range, vec.tree),
        }
    }

    /// Inserts the given item at the given index in this [`PersistentVec`].
    ///
    /// This method takes O(log n) time, where n is the length of the [`PersistentVec`].
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentVec;
    ///
    /// let vec = PersistentVec::from(vec![0, 2]);
    /// let vec = vec.insert(1, 1);
    /// assert_eq!(vec, PersistentVec::from(vec![0, 1, 2]))
    /// ```
    pub fn insert(self, index: usize, item: T) -> Self {
        Self {
            tree: self.tree.insert(index, item),
        }
    }

    /// Removes an item at the given index from this [`PersistentVec`].
    ///
    /// This method takes O(log n) time, where n is the length of the [`PersistentVec`].
    ///
    /// # Panics
    ///
    /// Panics if `index > self.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentVec;
    ///
    /// let vec = PersistentVec::from(vec![0, 1, 2]);
    /// let vec = vec.remove(1);
    /// assert_eq!(vec, PersistentVec::from(vec![0, 2]));
    /// ```
    pub fn remove(self, index: usize) -> Self {
        Self {
            tree: self.tree.remove(index),
        }
    }

    /// Splits this [`PersistentVec`] into two at the given index.
    ///
    /// This method takes O(log(n)) time, where `n` is the length of the [`PersistentVec`].
    ///
    /// # Panics
    ///
    /// Panics if `index > self.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentVec;
    ///
    /// let vec = PersistentVec::from(vec![0, 1, 2, 3]);
    /// let (vec_0, vec_1) = vec.split(2);
    /// assert_eq!(vec_0, PersistentVec::from(vec![0, 1]));
    /// assert_eq!(vec_1, PersistentVec::from(vec![2, 3]));
    /// ```
    pub fn split(self, index: usize) -> (Self, Self) {
        let (tree_0, tree_1) = self.tree.split(index);
        (Self { tree: tree_0 }, Self { tree: tree_1 })
    }

    /// Removes all items before the given index from this [`PersistentVec`].
    ///
    /// # Panics
    ///
    /// Panics if `index > self.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentVec;
    ///
    /// let vec = PersistentVec::from(vec![0, 1, 2, 3]);
    /// let vec = vec.truncate_before(2);
    /// assert_eq!(vec, PersistentVec::from(vec![2, 3]));
    /// ```
    pub fn truncate_before(self, index: usize) -> Self {
        Self {
            tree: self.tree.truncate_before(index),
        }
    }

    /// Removes all items after the given index from this [`PersistentVec`].
    ///
    /// # Panics
    ///
    /// Panics if `index > self.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentVec;
    ///
    /// let vec = PersistentVec::from(vec![0, 1, 2, 3]);
    /// let vec = vec.truncate_after(2);
    /// assert_eq!(vec, PersistentVec::from(vec![0, 1]));
    /// ```
    pub fn truncate_after(self, index: usize) -> Self {
        Self {
            tree: self.tree.truncate_after(index),
        }
    }

    /// Pops an item from the front of this [`PersistentVec`].
    ///
    /// This method takes O(log(n)) time, where `n` is the length of the [`PersistentVec`].
    ///
    /// If the [`PersistentVec`] is empty, this method has no effect.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentVec;
    ///
    /// let vec = PersistentVec::new();
    /// assert_eq!(vec.clone().pop_front(), vec);
    /// let vec = vec.push_front(0);
    /// let vec = vec.push_front(1);
    /// let vec = vec.pop_front();
    /// assert_eq!(vec.front(), Some(&0));
    /// ```
    pub fn pop_front(self) -> Self {
        Self {
            tree: self.tree.pop_front(),
        }
    }

    /// Pops an item from the back of this [`PersistentVec`].
    ///
    /// This method takes O(log(n)) time, where `n` is the length of the [`PersistentVec`].
    ///
    /// If the [`PersistentVec`] is empty, this method has no effect.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentVec;
    ///
    /// let vec = PersistentVec::new();
    /// assert_eq!(vec.clone().pop_back(), vec);
    /// let vec = vec.push_back(0);
    /// let vec = vec.push_back(1);
    /// let vec = vec.pop_back();
    /// assert_eq!(vec.back(), Some(&0));
    /// ```
    pub fn pop_back(self) -> Self {
        Self {
            tree: self.tree.pop_back(),
        }
    }

    /// Pushes an item to the front of this [`PersistentVec`].
    ///
    /// This method takes O(log(n)) time, where `n` is the length of the [`PersistentVec`].
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentVec;
    ///
    /// let vec = PersistentVec::new();
    /// let vec = vec.push_front(0);
    /// let vec = vec.push_front(1);
    /// assert_eq!(vec.front(), Some(&1));
    /// ```
    pub fn push_front(self, item: T) -> Self {
        Self {
            tree: self.tree.push_front(item),
        }
    }

    /// Pushes an item to the back of this [`PersistentVec`].
    ///
    /// This method takes O(log(n)) time, where `n` is the length of the [`PersistentVec`].
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentVec;
    ///
    /// let vec = PersistentVec::new();
    /// let vec = vec.push_back(0);
    /// let vec = vec.push_back(1);
    /// assert_eq!(vec.back(), Some(&1));
    /// ```
    pub fn push_back(self, item: T) -> Self {
        Self {
            tree: self.tree.push_back(item),
        }
    }
}

impl<T: Clone> Add for PersistentVec<T> {
    type Output = Self;

    /// Concatenates two [`PersistentVec`]s.
    ///
    /// This operation takes O(log(n)) time, where `n` is the sum of the lengths of the two
    /// [`PersistentVec`]s.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentVec;
    ///
    /// let vec_0 = PersistentVec::from(vec![0, 1]);
    /// let vec_1 = PersistentVec::from(vec![2, 3]);
    /// let vec = vec_0 + vec_1;
    /// assert_eq!(vec, PersistentVec::from(vec![0, 1, 2, 3]));
    /// ```
    fn add(self, other: Self) -> Self::Output {
        Self {
            tree: self.tree.concat(other.tree),
        }
    }
}

impl<T: Eq> Eq for PersistentVec<T> {}

impl<'a, T: Clone> From<&'a [T]> for PersistentVec<T> {
    /// Converts a `&[T]` to a [`PersistentVec`] by cloning the items.
    ///
    /// This function takes O(n * log(n)) time, where `n` is the length of the [`PersistentVec`].
    fn from(items: &'a [T]) -> Self {
        Self {
            tree: BTree::from(items),
        }
    }
}

impl<T: Clone> From<Vec<T>> for PersistentVec<T> {
    /// Converts an [`Vec`] to a [`PersistentVec`].
    ///
    /// This function takes O(n * log(n)) time, where `n` is the length of the [`PersistentVec`].
    fn from(items: Vec<T>) -> Self {
        Self {
            tree: BTree::from(items),
        }
    }
}

impl<T: Clone> FromIterator<T> for PersistentVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            tree: BTree::from_iter(iter),
        }
    }
}

impl<T> Index<usize> for PersistentVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).unwrap()
    }
}

impl<'a, T> IntoIterator for &'a PersistentVec<T> {
    type IntoIter = Iter<'a, T>;
    type Item = &'a T;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T: PartialEq> PartialEq for PersistentVec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.tree == other.tree
    }
}

/// An iterator over the items of a [`PersistentVec`].
///
/// This struct is created by [`PersistentVec::iter`].
pub struct Iter<'a, T> {
    iter: btree::Iter<'a, T, ()>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back()
    }
}

impl<'a, T> ExactSizeIterator for Iter<'a, T> {}

/// An iterator over a range of items in a [`PersistentVec`].
///
/// This struct is created by [`PersistentVec::range`].
pub struct Range<'a, T> {
    range: btree::Range<'a, T, ()>,
}

impl<'a, T> Iterator for Range<'a, T> {
    type Item = &'a T;

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }

    fn next(&mut self) -> Option<Self::Item> {
        self.range.next()
    }
}

impl<'a, T> DoubleEndedIterator for Range<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.range.next_back()
    }
}

impl<'a, T> ExactSizeIterator for Range<'a, T> {}
