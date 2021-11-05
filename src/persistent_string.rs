use {
    crate::{
        btree,
        btree::{Builder, BTree, Measured, Monoid},
        utf8, util,
    },
    std::{
        iter::FromIterator,
        ops::{Add, RangeBounds},
    },
};

/// A persistent string.
#[derive(Clone, Debug)]
pub struct PersistentString {
    tree: BTree<u8, CharBoundaryCount>,
}

impl PersistentString {
    /// Creates a new empty [`PersistentString`].
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentString;
    ///
    /// let string = PersistentString::new();
    /// ```
    pub fn new() -> Self {
        Self { tree: BTree::new() }
    }

    /// Replaces the given range of characters in this [`PersistentString`] with the given
    /// [`PersistentString`].
    ///
    /// This method takes O(log n) time, where n is the sum of the lengths of the two
    /// [`PersistentString`]s.
    ///
    /// # Panics
    ///
    /// Panics if the start of the range is greater than the end of the range, or the end of the
    /// range is greater than the length of the [`PersistentString`].
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentString;
    ///
    /// let string_0 = PersistentString::from("abcd");
    /// let string_1 = PersistentString::from("ef");
    /// let string = string_0.splice(1..3, string_1);
    /// assert_eq!(string, PersistentString::from("aefd"));
    /// ```
    pub fn splice<R: RangeBounds<usize>>(self, range: R, vec: Self) -> Self {
        let range = util::check_bounds(range, self.len());
        let start = self.byte_index(range.start);
        let end = self.byte_index(range.end);
        Self {
            tree: self.tree.splice(start..end, vec.tree),
        }
    }

    /// Inserts the given character at the given index in this [`PersistentString`].
    ///
    /// This method takes O(log n) time, where n is the length of the [`PersistentString`].
    ///
    /// # Panics
    ///
    /// Panics if `index > self.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentString;
    ///
    /// let string = PersistentString::from("ac");
    /// let string = string.insert(1, 'b');
    /// assert_eq!(string, PersistentString::from("abc"));
    /// ```
    ///
    pub fn insert(self, index: usize, ch: char) -> Self {
        assert!(index <= self.len());
        let index = self.byte_index(index);
        let mut bytes = [0; 4];
        Self {
            tree: self.tree.splice(
                index..index,
                BTree::<u8, CharBoundaryCount>::from(ch.encode_utf8(&mut bytes).as_bytes()),
            ),
        }
    }

    /// Removes a character ast the given index from this [`PersistentString`].
    ///
    /// This method takes O(log n) time, where n is the length of the [`PersistentString`].
    ///
    /// # Panics
    ///
    /// Panics if `index >= self.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentString;
    ///
    /// let string = PersistentString::from("abc");
    /// let string = string.remove(1);
    /// assert_eq!(string, PersistentString::from("ac"));
    /// ```
    pub fn remove(self, index: usize) -> Self {
        assert!(index < self.len());
        let index = self.byte_index(index);
        let mut bytes = self.tree.range(index..);
        let code_point = utf8::next_code_point(&mut bytes).unwrap();
        let ch = char::from_u32(code_point).unwrap();
        Self {
            tree: self.tree.splice(index..index + ch.len_utf8(), BTree::new()),
        }
    }

    /// Splits this [`PersistentString`] into two at the given index.
    ///
    /// This method takes O(log(n)) time, where `n` is the length of the [`PersistentString`].
    ///
    /// # Panics
    ///
    /// Panics if `index > self.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentString;
    ///
    /// let string = PersistentString::from("abcd");
    /// let (string_0, string_1) = string.split(2);
    /// assert_eq!(string_0, PersistentString::from("ab"));
    /// assert_eq!(string_1, PersistentString::from("cd"));
    /// ```
    pub fn split(self, index: usize) -> (Self, Self) {
        assert!(index <= self.len());
        let index = self.byte_index(index);
        let (tree_0, tree_1) = self.tree.split(index);
        (Self { tree: tree_0 }, Self { tree: tree_1 })
    }

    /// Removes all characters before the given index from this [`PersistentString`].
    ///
    /// # Panics
    ///
    /// Panics if `index > self.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentString;
    ///
    /// let string = PersistentString::from("abcd");
    /// let string = string.truncate_before(2);
    /// assert_eq!(string, PersistentString::from("cd"));
    /// ```
    pub fn truncate_before(self, index: usize) -> Self {
        assert!(index <= self.len());
        let index = self.byte_index(index);
        Self {
            tree: self.tree.truncate_before(index),
        }
    }

    /// Removes all characters after the given index from this [`PersistentString`].
    ///
    /// # Panics
    ///
    /// Panics if `index > self.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentString;
    ///
    /// let string = PersistentString::from("abcd");
    /// let string = string.truncate_after(2);
    /// assert_eq!(string, PersistentString::from("ab"));
    /// ```
    pub fn truncate_after(self, index: usize) -> Self {
        assert!(index <= self.len());
        let index = self.byte_index(index);
        Self {
            tree: self.tree.truncate_after(index),
        }
    }

    /// Pops a character from the front of this [`PersistentString`].
    ///
    /// This method takes O(log(n)) time, where `n` is the length of the [`PersistentString`].
    ///
    /// If the [`PersistentString`] is empty, this method has no effect.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentString;
    ///
    /// let string = PersistentString::new();
    /// assert_eq!(string.clone().pop_front(), string);
    /// let string = string.push_front('a');
    /// let string = string.push_front('b');
    /// let string = string.pop_front();
    /// assert_eq!(string.front(), Some('a'));
    /// ```
    pub fn pop_front(self) -> Self {
        match self.front() {
            Some(ch) => Self {
                tree: self.tree.truncate_before(ch.len_utf8()),
            },
            None => self,
        }
    }

    /// Pops a character from the back of this [`PersistentString`].
    ///
    /// This method takes O(log(n)) time, where `n` is the length of the [`PersistentString`].
    ///
    /// If the [`PersistentString`] is empty, this method has no effect.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentString;
    ///
    /// let string = PersistentString::new();
    /// assert_eq!(string.clone().pop_back(), string);
    /// let string = string.push_back('a');
    /// let string = string.push_back('b');
    /// let string = string.pop_back();
    /// assert_eq!(string.back(), Some('a'));
    /// ```
    pub fn pop_back(self) -> Self {
        match self.back() {
            Some(ch) => {
                let len = self.tree.len();
                Self {
                    tree: self.tree.truncate_after(len - ch.len_utf8()),
                }
            }
            None => self,
        }
    }

    /// Pushes a character to the front of this [`PersistentString`].
    ///
    /// This method takes O(log(n)) time, where `n` is the length of the [`PersistentString`].
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentString;
    ///
    /// let string = PersistentString::new();
    /// let string = string.push_front('a');
    /// let string = string.push_front('b');
    /// assert_eq!(string.front(), Some('b'));
    /// ```
    pub fn push_front(self, ch: char) -> Self {
        let mut bytes = [0; 4];
        Self {
            tree: ch
                .encode_utf8(&mut bytes)
                .as_bytes()
                .iter()
                .rev()
                .fold(self.tree, |tree, byte| tree.push_front(*byte)),
        }
    }

    /// Pushes a character to the back of this [`PersistentString`].
    ///
    /// This method takes O(log(n)) time, where `n` is the length of the [`PersistentString`].
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentString;
    ///
    /// let string = PersistentString::new();
    /// let string = string.push_back('a');
    /// let string = string.push_back('b');
    /// assert_eq!(string.back(), Some('b'));
    /// ```
    pub fn push_back(self, ch: char) -> Self {
        let mut bytes = [0; 4];
        Self {
            tree: ch
                .encode_utf8(&mut bytes)
                .as_bytes()
                .iter()
                .fold(self.tree, |tree, byte| tree.push_back(*byte)),
        }
    }

    /// Returns `true` if this string is [`PersistentString`].
    ///
    /// This method takes O(1) time.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentString;
    ///
    /// let string = PersistentString::new();
    /// assert!(string.is_empty());
    /// let string = string.push_back('a');
    /// assert!(!string.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the length of this [`PersistentString`].
    ///
    /// This method takes O(1) time.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentString;
    ///
    /// let string = PersistentString::new();
    /// assert_eq!(string.len(), 0);
    /// let string = string.push_back('a');
    /// assert_eq!(string.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.tree.tag().0
    }

    /// Returns the character at the front of this [`PersistentString`], or [`None`] if the
    /// [`PersistentString`] is empty.
    ///
    /// This method takes O(log(n)) time, where `n` is the length of the [`PersistentString`].
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentString;
    ///
    /// let string = PersistentString::new();
    /// assert_eq!(string.front(), None);
    /// let string = string.push_front('a');
    /// let string = string.push_front('b');
    /// assert_eq!(string.front(), Some('b'));
    /// ```
    pub fn front(&self) -> Option<char> {
        let mut bytes = self.tree.iter();
        let code_point = utf8::next_code_point(&mut bytes)?;
        Some(char::from_u32(code_point).unwrap())
    }

    /// Returns the character at the back of this [`PersistentString`], or [`None`] if the
    /// [`PersistentString`] is empty.
    ///
    /// This method takes O(log(n)) time, where `n` is the length of the [`PersistentString`].
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentString;
    ///
    /// let string = PersistentString::new();
    /// assert_eq!(string.back(), None);
    /// let string = string.push_back('a');
    /// let string = string.push_back('b');
    /// assert_eq!(string.back(), Some('b'));
    /// ```
    pub fn back(&self) -> Option<char> {
        let mut bytes = self.tree.iter();
        let code_point = utf8::next_code_point_reverse(&mut bytes)?;
        Some(char::from_u32(code_point).unwrap())
    }

    /// Returns the character at the given index in this [`PersistentString`], or [`None`] if
    /// `index >= self.len()`.
    ///
    /// This method takes O(log(n)) time, where `n` is the length of the string.
    ///
    /// ```
    /// use persistent_sequences::PersistentString;
    ///
    /// let string = PersistentString::from("abc");
    /// assert_eq!(string.get(1), Some('b'));
    /// ```
    pub fn get(&self, index: usize) -> Option<char> {
        if index >= self.len() {
            return None;
        }
        let index = self.byte_index(index);
        let mut bytes = self.tree.range(index..);
        let code_point = utf8::next_code_point(&mut bytes).unwrap();
        Some(char::from_u32(code_point).unwrap())
    }

    /// Returns an iterator over the characters of this [`PersistentString`].
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentString;
    ///
    /// let string = PersistentString::from("ab");
    /// let mut iter = string.iter();
    /// assert_eq!(iter.next(), Some('a'));
    /// assert_eq!(iter.next(), Some('b'));
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter(&self) -> Iter<'_> {
        Iter {
            count: self.len(),
            iter: self.tree.iter(),
        }
    }

    /// Returns an iterator over the given range of items in this [`PersistentString`].
    ///
    /// # Panics
    ///
    /// Panics if the start of the range is greater than the end of the range, or the end of the
    /// range is greater than the length of the [`PersistentString`].
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentString;
    ///
    /// let string = PersistentString::from("abcd");
    /// let mut range = string.range(1..3);
    /// assert_eq!(range.next(), Some('b'));
    /// assert_eq!(range.next(), Some('c'));
    /// assert_eq!(range.next(), None);
    /// ```
    pub fn range<R: RangeBounds<usize>>(&self, range: R) -> Range<'_> {
        let range = util::check_bounds(range, self.len());
        let start = self.byte_index(range.start);
        let end = self.byte_index(range.end);
        Range {
            count: range.end - range.start,
            range: self.tree.range(start..end),
        }
    }

    fn byte_index(&self, index: usize) -> usize {
        self.tree
            .search(|count| index < count.0)
            .unwrap_or_else(|| self.tree.len())
    }
}

impl Add for PersistentString {
    type Output = Self;

    /// Concatenates two [`PersistentString`]s.
    ///
    /// This operation takes O(log(n)) time, where `n` is the sum of the lengths of the two
    /// [`PersistentString`]s.
    ///
    /// # Examples
    ///
    /// ```
    /// use persistent_sequences::PersistentString;
    ///
    /// let string_0 = PersistentString::from("ab");
    /// let string_1 = PersistentString::from("cd");
    /// let string = string_0 + string_1;
    /// assert_eq!(string, PersistentString::from("abcd"));
    /// ```
    fn add(self, other: Self) -> Self::Output {
        Self {
            tree: self.tree.concat(other.tree),
        }
    }
}

impl Eq for PersistentString {}

impl<'a> From<&'a str> for PersistentString {
    /// Converts a `&str` to a [`PersistentString`].
    ///
    /// This function takes O(n * log(n)) time, where `n` is the length of the [`PersistentString`].
    fn from(string: &'a str) -> Self {
        Self {
            tree: BTree::from(string.as_bytes()),
        }
    }
}

impl From<String> for PersistentString {
    /// Converts a [`String`] to a [`PersistentString`].
    ///
    /// This function takes O(n * log(n)) time, where `n` is the length of the [`PersistentString`].
    fn from(string: String) -> Self {
        Self {
            tree: BTree::from(string.into_bytes()),
        }
    }
}

impl FromIterator<char> for PersistentString {
    fn from_iter<I: IntoIterator<Item = char>>(iter: I) -> Self {
        let mut builder = Builder::new();
        for ch in iter.into_iter() {
            let mut bytes = [0; 4];
            for &byte in ch.encode_utf8(&mut bytes).as_bytes() {
                builder.push(byte);
            }
        }
        Self {
            tree: builder.build()
        }
    }
}

impl<'a> IntoIterator for &'a PersistentString {
    type IntoIter = Iter<'a>;
    type Item = char;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl PartialEq for PersistentString {
    fn eq(&self, other: &Self) -> bool {
        self.tree == other.tree
    }
}

#[derive(Clone, Debug)]
struct CharBoundaryCount(usize);

impl Measured<u8> for CharBoundaryCount {
    fn measure(byte: &u8) -> Self {
        Self(if *byte as i8 >= -0x40 { 1 } else { 0 })
    }
}

impl Monoid for CharBoundaryCount {
    fn empty() -> Self {
        Self(0)
    }

    fn append(self: Self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

/// An iterator over the characters of a [`PersistentString`].
///
/// This struct is created by [`PersistentString::iter`].
pub struct Iter<'a> {
    count: usize,
    iter: btree::Iter<'a, u8, CharBoundaryCount>,
}

impl<'a> Iterator for Iter<'a> {
    type Item = char;

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.count, Some(self.count))
    }

    fn next(&mut self) -> Option<Self::Item> {
        if self.count == 0 {
            return None;
        }
        self.count -= 1;
        let code_point = utf8::next_code_point(&mut self.iter).unwrap();
        Some(char::from_u32(code_point).unwrap())
    }
}

impl<'a> DoubleEndedIterator for Iter<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.count == 0 {
            return None;
        }
        self.count -= 1;
        let code_point = utf8::next_code_point_reverse(&mut self.iter).unwrap();
        Some(char::from_u32(code_point).unwrap())
    }
}

impl<'a> ExactSizeIterator for Iter<'a> {}

/// An iterator over a range of characters of a [`PersistentString`].
///
/// This struct is created by [`PersistentString::range`].
pub struct Range<'a> {
    count: usize,
    range: btree::Range<'a, u8, CharBoundaryCount>,
}

impl<'a> Iterator for Range<'a> {
    type Item = char;

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.count, Some(self.count))
    }

    fn next(&mut self) -> Option<Self::Item> {
        if self.count == 0 {
            return None;
        }
        self.count -= 1;
        let code_point = utf8::next_code_point(&mut self.range).unwrap();
        Some(char::from_u32(code_point).unwrap())
    }
}

impl<'a> DoubleEndedIterator for Range<'a> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.count == 0 {
            return None;
        }
        self.count -= 1;
        let code_point = utf8::next_code_point_reverse(&mut self.range).unwrap();
        Some(char::from_u32(code_point).unwrap())
    }
}

impl<'a> ExactSizeIterator for Range<'a> {}

#[cfg(test)]
mod tests {
    use {super::*, proptest::prelude::*};

    proptest! {
        #[test]
        fn splice(
            ((string_0, (start, end)), string_1) in (
                ".*"
                    .prop_map(|string| PersistentString::from(string))
                    .prop_flat_map(|tree| {
                        let len = tree.len();
                        (
                            Just(tree),
                            (0..=len).prop_flat_map(move |start| (Just(start), start..=len))
                        )
                    }),
                ".*".prop_map(|string| PersistentString::from(string))
            )
        ) {
            let new_string = string_0.clone().splice(start..end, string_1.clone());
            assert_eq!(
                new_string.iter().collect::<String>(),
                string_0
                    .iter()
                    .take(start)
                    .chain(string_1.iter()).chain(string_0.iter().skip(end))
                    .collect::<String>()
            );
        }

        #[test]
        fn insert(
            ((string, index), ch) in (
                ".*"
                    .prop_map(|string| PersistentString::from(string))
                    .prop_flat_map(|tree| {
                        let len = tree.len();
                        (Just(tree), 0..=len)
                    }),
                any::<char>()
            )
        ) {
            use std::iter;

            let new_string = string.clone().insert(index, ch);
            assert_eq!(
                new_string.iter().collect::<String>(),
                string
                    .iter()
                    .take(index)
                    .chain(iter::once(ch))
                    .chain(string.iter().skip(index))
                    .collect::<String>()
            )
        }

        #[test]
        fn remove(
            (string, index) in "..*"
                .prop_map(|string| PersistentString::from(string))
                .prop_flat_map(|tree| {
                    let len = tree.len();
                    (Just(tree), 0..len)
                })
        ) {
            let new_string = string.clone().remove(index);
            assert_eq!(
                new_string.iter().collect::<String>(),
                string
                    .iter()
                    .take(index)
                    .chain(string.iter().skip(index + 1))
                    .collect::<String>()
            )
        }

        #[test]
        fn split(
            (string, index) in ".*"
                .prop_map(|string| PersistentString::from(string))
                .prop_flat_map(|string| {
                    let len = string.len();
                    (Just(string), 0..=len)
                })
        ) {
            let (string_0, string_1) = string.clone().split(index);
            assert_eq!(
                string_0.iter().collect::<String>(),
                string.iter().take(index).collect::<String>()
            );
            assert_eq!(
                string_1.iter().collect::<String>(),
                string.iter().skip(index).collect::<String>()
            );
        }

        #[test]
        fn truncate_before(
            (string, index) in ".*"
                .prop_map(|string| PersistentString::from(string))
                .prop_flat_map(|string| {
                    let len = string.len();
                    (Just(string), 0..=len)
                })
        ) {
            let new_string = string.clone().truncate_before(index);
            assert_eq!(
                new_string.iter().collect::<String>(),
                string.iter().skip(index).collect::<String>()
            );
        }

        #[test]
        fn truncate_after(
            (string, index) in ".*"
                .prop_map(|string| PersistentString::from(string))
                .prop_flat_map(|string| {
                    let len = string.len();
                    (Just(string), 0..=len)
                })
        ) {
            let new_string = string.clone().truncate_after(index);
            assert_eq!(
                new_string.iter().collect::<String>(),
                string.iter().take(index).collect::<String>()
            );
        }

        #[test]
        fn pop_front(string in ".*".prop_map(|string| PersistentString::from(string))) {
            let new_string = string.clone().pop_front();
            assert_eq!(
                new_string.iter().collect::<String>(),
                string.iter().skip(1).collect::<String>()
            );
        }

        #[test]
        fn pop_back(string in ".*".prop_map(|string| PersistentString::from(string))) {
            let new_string = string.clone().pop_back();
            assert_eq!(
                new_string.iter().collect::<String>(),
                string.iter().take(string.len().saturating_sub(1)).collect::<String>()
            );
        }

        #[test]
        fn push_front(
            string in ".*".prop_map(|string| PersistentString::from(string)),
            ch in any::<char>()
        ) {
            use std::iter;

            let new_string = string.clone().push_front(ch);
            assert_eq!(
                new_string.iter().collect::<String>(),
                iter::once(ch).chain(string.iter()).collect::<String>()
            );
        }

        #[test]
        fn push_back(
            string in ".*".prop_map(|string| PersistentString::from(string)),
            ch in any::<char>()
        ) {
            use std::iter;

            let new_string = string.clone().push_back(ch);
            assert_eq!(
                new_string.iter().collect::<String>(),
                string.iter().chain(iter::once(ch)).collect::<String>()
            );
        }

        #[test]
        fn len(string in ".*".prop_map(|string| PersistentString::from(string))) {
            assert_eq!(string.len(), string.iter().count());
        }

        #[test]
        fn front(string in ".*".prop_map(|string| PersistentString::from(string))) {
            assert_eq!(string.front(), string.iter().next())
        }

        #[test]
        fn back(string in ".*".prop_map(|string| PersistentString::from(string))) {
            assert_eq!(string.back(), string.iter().next_back());
        }

        #[test]
        fn get((string, index) in "..*"
            .prop_map(|string| PersistentString::from(string))
            .prop_flat_map(|string| {
                let len = string.len();
                (Just(string), 0..len)
            })
        ) {
            assert_eq!(string.get(index), string.iter().nth(index));
        }

        #[test]
        fn iter(string in ".*") {
            let persistent_string = PersistentString::from(string.clone());
            assert_eq!(
                persistent_string.iter().collect::<String>(),
                string
            );
            assert_eq!(
                persistent_string.iter().rev().collect::<String>(),
                string.chars().rev().collect::<String>()
            )
        }

        #[test]
        fn range(
            (string, (start, end)) in ".*"
                .prop_map(|string| PersistentString::from(string))
                .prop_flat_map(|tree| {
                    let len = tree.len();
                    (
                        Just(tree),
                        (0..=len).prop_flat_map(move |start| (Just(start), start..=len))
                    )
                }),
        ) {
            assert_eq!(
                string.range(start..end).collect::<String>(),
                string.iter().skip(start).take(end - start).collect::<String>()
            );
            assert_eq!(
                string.range(start..end).rev().collect::<String>(),
                string.iter().skip(start).take(end - start).rev().collect::<String>()
            );
        }
    }
}
