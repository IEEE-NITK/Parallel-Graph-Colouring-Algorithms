# Parallel-Graph-Colouring-Algorithms
This repository contains all the work that is done under this project

The following methods have been implemented in CUDA:
- Vertex Based Iterative PGC  
    It is a simple approach in which the vertices are colored speculatively, and conflicts that occur due to race conditions are detected. The above steps are repeated until there are no conflicts.
- Edge Based PGC
    For each vertex v, VFORBIDDEN(v) holds the list of forbidden colors for v. First the algorithm goes through all vertices and picks the smallest available color for v based on VFORBIDDEN(v). Then conflicts are detected by going through all edges and checking if any two endpoints have the same color. If so, one endpoint is marked with conflict. Then VFORBIDDEN(v) is atomically based on colors of neighbours of v, updated by going through all edges. The algorithm terminates when there are no conflicts. The above steps are repeated until there are no conflicts.
