from __future__ import annotations

import cProfile
import time
from typing import Optional

from huffman import HuffmanTree
from utils import *
# ====================
# New data types


class HMNode:
    """A node of a linked list that stores a huffman tree with its
    corresponding frequency.

    === Attributes ===
    item:
        The Huffman tree stored in this node.
    freq:
        The corresponding frequency of the huffman tree.
    next:
        The next node in the list, or None if there are no more nodes.
    """
    item: HuffmanTree
    freq: int
    next: Optional[HMNode]

    def __init__(self, item: HuffmanTree, freq: int) -> None:
        """Initialize a new node storing <item>, with no next node.
        """
        self.item = item
        self.freq = freq
        self.next = None

    def merge_node(self, other: HMNode) -> HMNode:
        """Merge the other node into this node.
        """
        merged_tree = HuffmanTree(None, self.item, other.item)
        return HMNode(merged_tree, self.freq + other.freq)

    def insert_node(self, other: HMNode) -> None:
        """Insert the other node after this node.
        """
        self.next, other.next = other, self.next


class PointerHMNode:
    """A pointer to a node of a linked list that stores a huffman tree with its
    corresponding frequency.

    === Attributes ===
    node:
        The node that this pointer points to.
    """
    first: Optional[HMNode]

    def __init__(self, node: HMNode) -> None:
        """Initialize a new pointer to the given node.
        """
        self.first = node

    def insert_node(self, other: HMNode) -> None:
        """Insert a HMNode into the linked list in an inserting order of
        its frequency.

        Precondition: The first node is the smallest and sorted.
        """
        if other.freq < self.first.freq:
            other.next = self.first
            self.first = other
        else:
            current = self.first
            while (current.next is not None
                   and current.next.freq <= other.freq):
                current = current.next
            current.insert_node(other)

    def rev_insert_node(self, other: HMNode) -> None:
        """Does the exact same as insert node, but the nodes are added
        in descending order of its frequency
        """
        if other.freq >= self.first.freq:
            other.next = self.first
            self.first = other
        else:
            current = self.first
            while (current.next is not None
                   and current.next.freq > other.freq):
                current = current.next
            current.insert_node(other)

    def merge_node(self, other: HMNode) -> None:
        """Merge the other node into the first node.

        Precondition: The first node is the smallest and sorted.
        """
        a = self.first.merge_node(other)
        if self.first.next.next is None:
            self.first = a
        else:
            self.first = self.first.next.next
            self.first.previous = None
            self.insert_node(a)

    def pop(self) -> int:
        """ Removes the first element from a pointer and returns
        symbol of its leaf.

        Precondition: Self is not None
        """
        if self.first.next is not None:
            a = self.first
            self.first = self.first.next
            return a.item.symbol
        else:
            a = self.first.item
            self.first = None
            return a.symbol


# ====================
# Functions for compression


def build_frequency_dict(text: bytes) -> dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    d = {}
    for i in text:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    return d


def build_huffman_tree(freq_dict: dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    """
    if len(freq_dict) == 1:
        key = list(freq_dict.keys())[0]
        a = HuffmanTree(key)
        if key == 1:
            b = HuffmanTree(0)
        else:
            b = HuffmanTree(1)
        return HuffmanTree(None, b, a)
    else:
        pointer = _make_hm_linked_list(freq_dict)
        while pointer.first.next is not None:
            b = pointer.first.next
            pointer.merge_node(b)
        return pointer.first.item


def _make_hm_linked_list(freq_dict: dict[int, int]) -> PointerHMNode:
    """ Return a sorted list of input dict in a decreasing order
    of its frequency values.
    """
    keys = freq_dict.keys()
    temp_pointer = None
    for key in keys:
        temp_leaf = HuffmanTree(key)
        temp_node = HMNode(temp_leaf, freq_dict[key])
        if temp_pointer is None:
            temp_pointer = PointerHMNode(temp_node)
        else:
            temp_pointer.insert_node(temp_node)
    return temp_pointer


def get_codes(tree: HuffmanTree) -> dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    >>> a = {5: 16, 2: 47, 1: 59, 3: 40, 9: 3, 4: 37, 6: 15, 7: 9, 8: 7}
    >>> b = build_huffman_tree(a)
    >>> c = get_codes(b)
    >>> c == {1: "10", 2: "00", 3: "111", 4: "110", 5: "0111", \
    6: "0110", 7: "0100", 8: "01011", 9: "01010"}
    True
    """
    return _helper_get_codes(tree, "")


def _helper_get_codes(node: HuffmanTree, code: str,) -> dict[int, str]:
    """Helper function for get_code that uses conquer and divides method
    """
    if node.is_leaf():
        return {node.symbol: code}
    else:
        codes = {}
        if node.left:
            codes.update(_helper_get_codes(node.left, code + "0"))
        if node.right:
            codes.update(_helper_get_codes(node.right, code + "1"))
        return codes


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    _helper_number_nodes(tree, 0)


def _helper_number_nodes(tree: HuffmanTree, count: int) -> int:
    """ Helper function, that conquers and divides and records count
    and successfully numbers the tree.
    """
    if tree is None or tree.is_leaf():
        return count
    else:
        count = _helper_number_nodes(tree.left, count)
        count = _helper_number_nodes(tree.right, count)
        tree.number = count
        return count + 1


def avg_length(tree: HuffmanTree, freq_dict: dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    """
    ans = 0
    codes = get_codes(tree)
    for key in freq_dict:
        ans += len(codes[key]) * freq_dict[key]
    return ans / sum(freq_dict.values())


def compress_bytes(text: bytes, codes: dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    result = []
    bits = ""
    for byte in text:
        bits += codes[byte]
        while len(bits) >= 8:
            result.append(bits_to_byte(bits[:8]))
            bits = bits[8:]
    if bits:
        result.append(bits_to_byte(bits))
    return bytes(result)


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    >>> hm0 = HuffmanTree(None, HuffmanTree(104), HuffmanTree(101))
    >>> hm1 = HuffmanTree(None, HuffmanTree(119), HuffmanTree(114))
    >>> hm2 = HuffmanTree(None, hm0, hm1)
    >>> hm3 = HuffmanTree(None, HuffmanTree(100), HuffmanTree(111))
    >>> hm4 = HuffmanTree(None, HuffmanTree(108), hm3)
    >>> hm5 = HuffmanTree(None, hm2, hm4)
    >>> number_nodes(hm5)
    >>> list(tree_to_bytes(hm5))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111,\
    0, 108, 1, 3, 1, 2, 1, 4]
    """
    ans = []
    _helper_tree_to_bytes(tree, ans)
    return bytes(ans)


def _helper_tree_to_bytes(tree: HuffmanTree, ans: list) -> None:
    """ Does conquer and divide and helps tree_to_bytes function.
    """
    if tree.is_leaf():
        return
    else:
        _helper_tree_to_bytes(tree.left, ans)
        _helper_tree_to_bytes(tree.right, ans)
        if tree.left.is_leaf():
            x = [0, tree.left.symbol]
        else:
            x = [1, tree.left.number]
        if tree.right.is_leaf():
            y = [0, tree.right.symbol]
        else:
            y = [1, tree.right.number]
        ans.extend(x + y)


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree)
              + int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: list[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    """
    return _helper_generate_tree_general(node_lst, root_index)


def _helper_generate_tree_general(node_lst: list[ReadNode],
                                  index: int) -> HuffmanTree:
    """ Helper function for generate_tree_general.
    """
    node = node_lst[index]
    node.number = index
    if node.l_type == 0:
        left = HuffmanTree(node.l_data)
    else:
        left = _helper_generate_tree_general(node_lst, node.l_data)
    if node.r_type == 0:
        right = HuffmanTree(node.r_data)
    else:
        right = _helper_generate_tree_general(node_lst, node.r_data)
    return HuffmanTree(None, left, right)


def generate_tree_postorder(node_lst: list[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    """
    ans, _ = _helper_generate_tree_po(node_lst, root_index)
    return ans


def _helper_generate_tree_po(node_lst: list[ReadNode],
                             index: int) -> tuple[HuffmanTree, int]:
    """ Helper function for generate_tree_postorder.
    """
    tree = HuffmanTree(None, None, None)
    node = node_lst[index]
    tree.number = index
    if node.r_type == 0:
        tree.right = HuffmanTree(node.r_data)
    else:
        tree.right, index = _helper_generate_tree_po(node_lst, index - 1)
    if node.l_type == 0:
        tree.left = HuffmanTree(node.l_data)
    else:
        tree.left, index = _helper_generate_tree_po(node_lst, index - 1)
    return tree, index


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    >>> tree2 = build_huffman_tree(build_frequency_dict(b'holyshatsomeoneeillme'))
    >>> number_nodes(tree2)
    >>> decompress_bytes(tree2, \
                compress_bytes(b'holyshatsomeoneeillme', get_codes(tree2)), \
                len(b'holyshatsomeoneeillme'))
    b'holyshatsomeoneeillme'
    >>> tree3 = build_huffman_tree(build_frequency_dict(b'1122'))
    >>> number_nodes(tree3)
    >>> decompress_bytes(tree3, \
                compress_bytes(b'1122', get_codes(tree3)), len(b'1122'))
    b'1122'
    """
    temp_tree = tree
    ans = bytearray()
    for byte in text:
        if len(ans) == size:
            break
        bits = byte_to_bits(byte)
        for i in range(8):
            if bits[i] == "1":
                temp_tree = temp_tree.right
            else:
                temp_tree = temp_tree.left
            if temp_tree.is_leaf():
                ans.append(temp_tree.symbol)
                temp_tree = tree
            if len(ans) == size:
                break
    return bytes(ans)

def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def improve_tree(tree: HuffmanTree, freq_dict: dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)

    >>> avg_length(tree, freq)
    2.31
    """
    pointer = _make_rev_linked_list(freq_dict)
    queue = [tree]
    layer = []
    while queue or pointer.first is not None:
        layer.append(queue)
        current = queue.pop(0)
        if current.is_leaf():
            current.symbol = pointer.pop()
        else:
            queue.append(current.left)
            queue.append(current.right)


def _make_rev_linked_list(freq_dict: dict[int, int]) -> PointerHMNode:
    """ Return a sorted list of input dict in a decreasing order
    of its frequency values.
    """
    keys = freq_dict.keys()
    temp_pointer = None
    for key in keys:
        temp_leaf = HuffmanTree(key)
        temp_node = HMNode(temp_leaf, freq_dict[key])
        if temp_pointer is None:
            temp_pointer = PointerHMNode(temp_node)
        else:
            temp_pointer.rev_insert_node(temp_node)
    return temp_pointer


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['compress_file', 'decompress_file'],
        'allowed-import-modules': [
            'python_ta', 'doctest', 'typing', '__future__',
            'time', 'utils', 'huffman', 'random'
        ],
        'disable': ['W0401']
    })

    mode = input(
        "Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        #cProfile.run('compress_file(fname, fname + ".huf")')
        print(f"Compressed {fname} in {time.time() - start} seconds.")

    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        #cProfile.run('decompress_file(fname, fname + ".orig")')
        print(f"Decompressed {fname} in {time.time() - start} seconds.")
