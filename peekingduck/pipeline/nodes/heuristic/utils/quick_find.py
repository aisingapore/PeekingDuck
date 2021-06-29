# Copyright 2021 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
QuickFind algorithm used to connect nodes
"""

from typing import List


class QuickFind:
    """
    The general form of the quick-find algorithm that is used to connect nodes.
    How it works: https://www.cs.princeton.edu/~rs/AlgsDS07/01UnionFind.pdf
    Note that although there are faster algorithms such as union-find, we are using
    quick-find here because we need the returning array to have the final group numbers,
    and not have to continue to chase the roots of the nodes subsequently.
    """

    def __init__(self, arr_size: int) -> None:
        self.group_alloc = []
        self.arr_size = arr_size
        for i in range(arr_size):
            self.group_alloc.append(i)

    def get_group_alloc(self) -> List[int]:
        """Getter function for self.group_alloc

        Returns:
            (list): a list containing group numbers allocated to each node.
        """
        return self.group_alloc

    def union(self, node_1_idx: int, node_2_idx: int) -> None:
        """Unites two nodes and other nodes that have been connected to them.

        Args:
            node_1_idx (int): index of the first node
            node_2_idx (int): index of the second node
        """

        node_1_group = self.group_alloc[node_1_idx]
        node_2_group = self.group_alloc[node_2_idx]
        for i in range(self.arr_size):
            if self.group_alloc[i] == node_1_group:
                self.group_alloc[i] = node_2_group

    def connected(self, node_1_idx: int, node_2_idx: int) -> bool:
        """Checks if two nodes are connected.

        Args:
            node_1_idx (int): index of the first node
            node_2_idx (int): index of the second node

        Returns:
            (bool): True if 2 nodes are connected
        """
        return self.group_alloc[node_1_idx] == self.group_alloc[node_2_idx]
