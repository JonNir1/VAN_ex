import warnings
import numpy as np
import igraph as ig
from typing import Optional, List


class Graph:

    # _FrameIdx = "FrameIdx"
    # _Symbol = "Symbol"
    # _Weight = "weight"
    # _Covariance = "cov"

    def __init__(self, should_warn=False):
        self._graph = ig.Graph(0, directed=False)
        self._should_warn = should_warn

    def to_numpy(self) -> np.ndarray:
        adj_mat = self._graph.get_adjacency(attribute="weight")
        return np.array(adj_mat.data)

    def create_vertex(self, frame_idx: int, symbol: int) -> ig.Vertex:
        v = self._graph.add_vertex()
        v["FrameIdx"] = frame_idx
        v["Symbol"] = symbol
        return v

    def get_vertex_id(self, frame_idx: Optional[int] = None, symbol: Optional[int] = None) -> Optional[int]:
        # Returns the VertexID matching the provided FrameIdx or Symbol, or None if no such vertex exists
        # @throws a ValueError if both $frame_idx and $symbol are None
        if frame_idx is None and symbol is None:
            raise ValueError("Must provide FrameIdx or Symbol to search by.")
        if frame_idx is not None:
            vertex_ids = self._graph.vs.select(FrameIdx=frame_idx).indices
        else:  # symbol is not None
            vertex_ids = self._graph.vs.select(Symbol=symbol).indices
        if len(vertex_ids) == 0:
            return None
        return vertex_ids[0]

    def create_or_update_vertex(self, frame_idx: int, symbol: int) -> Optional[ig.Vertex]:
        """
        Finds a vertex based on it's FrameIdx or Frame symbol, and if no such vertex exists, creates one.
        Then updates the vertex to include the attributes FrameIdx and symbol with provided values.
        """
        # find vertex id using FrameIdx / Frame's symbol / create new vertex:
        v_id = self.get_vertex_id(frame_idx=frame_idx)
        if v_id is None:
            v_id = self.get_vertex_id(symbol=symbol)
        if v_id is None:
            v = self._graph.add_vertex()
            v_id = v.index
        if v_id is None:
            return None
        v = self._graph.vs[v_id]
        v["FrameIdx"] = frame_idx
        v["Symbol"] = symbol
        return v

    def create_or_update_edge(self, v1_id: int, v2_id: int, cov: Optional[np.ndarray]) -> ig.Edge:
        """
        Creates or updates existing edge between vertices $v1 and $v2, such that the edge will have
            weight that is the determinant of $cov (which should always be non-negative because it is a PSD matrix)
        @throws AssertionError if $cov is not a 6x6 matrix with non-negative determinant
        """
        assert cov.shape == (6, 6), f"Covariance matrix must be of shape 6x6, not {cov.shape}"
        det = np.linalg.det(cov)
        assert det >= 0, f"Covariance matrix must be PSD with non-negative determinant, not {det:.3f}"
        edge_id = self._graph.get_eid(v1_id, v2_id, error=False)
        if edge_id == -1:
            edge = self._graph.add_edge(v1_id, v2_id)
        else:
            edge = self._graph.es[edge_id]
        edge["cov"] = cov
        edge["weight"] = det
        return edge

    def get_relative_covariance(self, source: int, target: int) -> np.ndarray:
        # Returns the sum of all covariances along the shortest path
        path_edge_ids = self._get_shortest_path(source, target)
        edge_covs = self._graph.es[path_edge_ids]["cov"]
        return sum(edge_covs)

    def _get_shortest_path(self, source: int, target: int) -> Optional[List[int]]:
        # Returns a list of Edge IDs if a path exists, or None if no path is available
        if self._should_warn:
            return self.__get_shortest_path_impl(source, target)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.__get_shortest_path_impl(source, target)

    def __get_shortest_path_impl(self, source: int, target: int) -> Optional[List[int]]:
        """
        Returns a list of Edge IDs if a path exists, or None if no path is available
        @ Warns if the source and target vertices are not connected
        see documentation: https://igraph.org/python/tutorial/develop/tutorials/shortest_paths/shortest_paths.html
        """
        w = self._graph.es["weight"]
        res = self._graph.get_shortest_paths(source, target, weights=w, output="epath")
        edge_ids = res[0]
        if len(edge_ids) == 0:
            return None
        return edge_ids

