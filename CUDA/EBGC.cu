/*
 *   CUDA C/C++ implementation for Parallel Graph Coloring for Manycore Architectures
 *   {@link https://ieeexplore.ieee.org/abstract/document/7516086}
 *
 *   @author Ashwin Joisa
 *   @author Shanthanu Rai
 *   @author Rohit MP
 */

// Include header files
#include <bits/stdc++.h>
#include <cuda.h>

#define MAX_THREADS 1024
#define CEIL(a, b) ((a - 1) / b + 1)

using namespace std;

float device_time_taken;

// Catch Cuda errors
void catchCudaError(cudaError_t error, const char *function)
{
    if (error != cudaSuccess)
    {
        printf("\n====== Cuda Error Code %i ======\n %s in CUDA %s\n", error, cudaGetErrorString(error), function);
        exit(-1);
    }
}

// Host Memory
void readGraph(int &nodeCount, int &edgeCount,
    int &maxDegree, int **adjacencyList, int **adjacencyListPointers, 
    int **edgeListX, int **edgeListY) {

    int u, v;
    cin >> nodeCount >> edgeCount;

    // edge (u, v) => edgeListX[i] = u, edgeListY[i] = v
    *edgeListX = new int[edgeCount];
    *edgeListY = new int[edgeCount];

    // Use vector of vectors temporarily to input graph
    vector<int> *adj = new vector<int>[nodeCount];
    for (int i = 0; i < edgeCount; i++) {
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);

        (*edgeListX)[i] = u;
        (*edgeListY)[i] = v;
    }

    // Copy into compressed adjacency List
    *adjacencyListPointers = new int[nodeCount +1];
    *adjacencyList = new int[2 * edgeCount +1];

    int pos = 0;
    for(int i=0; i<nodeCount; i++) {
        (*adjacencyListPointers)[i] = pos;
        for(int node : adj[i])
            (*adjacencyList)[pos++] = node;
    }
    (*adjacencyListPointers)[nodeCount] = pos;

    // Calculate max degree
    maxDegree = INT_MIN;
    for(int i=0; i<nodeCount; i++)
        maxDegree = max(maxDegree, (int)adj[i].size());

    delete[] adj;
}

__global__ void assignColours(int nodeCount, int edgeCount, int *adjacencyList, int *adjacencyListPointers, 
    int *edgeListX, int *edgeListY, int *coloured, int *CS, int *colour, int *VForbidden) {

    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if(id < nodeCount) {
        coloured[0] = 0;
    }
    __syncthreads();
    // ################################################ Assign Colours
    if(id < nodeCount && colour[id] == 0) {
        if(~(VForbidden[id])) {
            // Some bit is zero => Some colour available
            // colour = First available bit (zero bit)
            int temp = (~(VForbidden[id]));
            colour[id] = (temp & -temp);
        }
        else {
            CS[id]++;
            VForbidden[id] = 0;
        }
    }
}
        
__global__ void detectConflicts(int nodeCount, int edgeCount, int *adjacencyList, int *adjacencyListPointers, 
    int *edgeListX, int *edgeListY, int *coloured, int *CS, int *colour, int *VForbidden) {

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    
    // ################################################ Detect Conflicts
    if(id < edgeCount) {
        int u = edgeListX[id], v = edgeListY[id];
        if(CS[u] == CS[v]) {
            if(u < v) {
                atomicCAS(&colour[u], colour[v], 0);
                if(colour[u] == 0) {
                    *coloured = 1;
                }
            }
            else {
                atomicCAS(&colour[v], colour[u], 0);
                if(colour[v] == 0) {
                    *coloured = 1;
                }
            }

            __syncthreads();
            // ################################################ VForbidden

            if(colour[u] && colour[v] == 0) 
                atomicOr(&VForbidden[v], colour[u]);
            if(colour[v] && colour[u] == 0) 
                atomicOr(&VForbidden[u], colour[v]);
        }
    }
}

__global__ void convertColour(int nodeCount, int edgeCount, int *adjacencyList, int *adjacencyListPointers, 
    int *edgeListX, int *edgeListY, int *coloured, int *CS, int *colour, int *VForbidden) {
    
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(id < nodeCount) {
        int cnt = 0, c = colour[id];
        while(c) {
            c >>= 1;
            cnt++;
        }
        colour[id] = CS[id] * 32 + cnt;
    }
}

int *graphColouring(int nodeCount, int edgeCount, int *device_adjacencyList, int *device_adjacencyListPointers, 
    int *device_edgeListX, int *device_edgeListY, int maxDegree) {

    int *host_colours = new int[nodeCount], *done = new int;
    int *device_colours, *device_coloured;
    int *device_CS, *device_VForbidden;

    catchCudaError(cudaMalloc((void **)&device_colours, sizeof(int) * nodeCount), "Malloc - Colours");
    catchCudaError(cudaMalloc((void **)&device_CS, sizeof(int) * nodeCount), "Malloc - CS");
    catchCudaError(cudaMalloc((void **)&device_VForbidden, sizeof(int) * nodeCount), "Malloc - VForbidden");
    catchCudaError(cudaMalloc((void **)&device_coloured, sizeof(int)), "Malloc - Coloured");
    
    // Timer
    cudaEvent_t device_start, device_end;
    catchCudaError(cudaEventCreate(&device_start), "Event Create");
    catchCudaError(cudaEventCreate(&device_end), "Event Create");
    catchCudaError(cudaEventRecord(device_start), "Event Record");

    do {
        assignColours <<< CEIL(max(nodeCount, edgeCount), MAX_THREADS), MAX_THREADS >>> (nodeCount, edgeCount, device_adjacencyList,
            device_adjacencyListPointers, device_edgeListX, device_edgeListY, device_coloured, device_CS, device_colours, device_VForbidden);
        detectConflicts <<< CEIL(max(nodeCount, edgeCount), MAX_THREADS), MAX_THREADS >>> (nodeCount, edgeCount, device_adjacencyList,
            device_adjacencyListPointers, device_edgeListX, device_edgeListY, device_coloured, device_CS, device_colours, device_VForbidden);
        
        catchCudaError(cudaMemcpy(&done, device_coloured, sizeof(int), cudaMemcpyDeviceToHost), "Memcpy00");
    }while(done);

    convertColour <<< CEIL(max(nodeCount, edgeCount), MAX_THREADS), MAX_THREADS >>> (nodeCount, edgeCount, device_adjacencyList, 
        device_adjacencyListPointers, device_edgeListX, device_edgeListY, device_coloured, device_CS, device_colours, device_VForbidden);

    cudaThreadSynchronize();
    // Timer
    catchCudaError(cudaEventRecord(device_end), "Event Record");
    catchCudaError(cudaEventSynchronize(device_end), "Event Synchronize");
    catchCudaError(cudaEventElapsedTime(&device_time_taken, device_start, device_end), "Elapsed time");

    // Copy colours to host and return
    catchCudaError(cudaMemcpy(host_colours, device_colours, sizeof(int) * nodeCount, cudaMemcpyDeviceToHost), "Memcpy3");

    delete[] host_conflicts;
    catchCudaError(cudaFree(device_colours), "Free");
    catchCudaError(cudaFree(device_coloured), "Free");
    catchCudaError(cudaFree(device_CS), "Free");
    catchCudaError(cudaFree(device_VForbidden), "Free");
    catchCudaError(cudaFree(device_conflicts), "Free");

    return host_colours;
}

int main(int argc, char *argv[]) {

    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <graph_input_file> [output_file]\n";
        return 0;
    }

    int nodeCount, edgeCount, maxDegree;
    int *adjacencyList = NULL, *adjacencyListPointers = NULL, *edgeListX = NULL, *edgeListY = NULL;
    int *device_adjacencyList, *device_adjacencyListPointers, *device_edgeListX, *device_edgeListY;

    char choice = 'n';
    cout << "Would you like to print the colouring of the graph? (y/n) ";
    cin >> choice;

    freopen(argv[1], "r", stdin);
    readGraph(nodeCount, edgeCount, maxDegree, &adjacencyList, &adjacencyListPointers, &edgeListX, &edgeListY);
    
    // Alocate device memory and copy - Adjacency List
    catchCudaError(cudaMalloc((void **)&device_adjacencyList, sizeof(int) * (2 * edgeCount + 1)), "Malloc5");
    catchCudaError(cudaMemcpy(device_adjacencyList, adjacencyList, sizeof(int) * (2 * edgeCount + 1), cudaMemcpyHostToDevice), "Memcpy11");

    // Alocate device memory and copy - Adjacency List Pointers
    catchCudaError(cudaMalloc((void **)&device_adjacencyListPointers, sizeof(int) * (nodeCount + 1)), "Malloc6");
    catchCudaError(cudaMemcpy(device_adjacencyListPointers, adjacencyListPointers, sizeof(int) * (nodeCount + 1), cudaMemcpyHostToDevice), "Memcpy12");

    // Alocate device memory and copy - Edge List X
    catchCudaError(cudaMalloc((void **)&device_edgeListX, sizeof(int) * (edgeCount)), "Malloc - Edge List X");
    catchCudaError(cudaMemcpy(device_edgeListX, edgeListX, sizeof(int) * (edgeCount), cudaMemcpyHostToDevice), "Memcpy - Edge List X");

    // Alocate device memory and copy - Edge List Y
    catchCudaError(cudaMalloc((void **)&device_edgeListY, sizeof(int) * (edgeCount)), "Malloc - Edge List Y");
    catchCudaError(cudaMemcpy(device_edgeListY, edgeListY, sizeof(int) * (edgeCount), cudaMemcpyHostToDevice), "Memcpy - Edge List Y");

    int *colouring = graphColouring(nodeCount, edgeCount, device_adjacencyList, device_adjacencyListPointers, 
        device_edgeListX, device_edgeListY, maxDegree);

    // calculating number of colours
    int chromaticNumber = INT_MIN;
    for (int i = 0; i < nodeCount; i++) {
        chromaticNumber = max(chromaticNumber, colouring[i]);
        if(choice == 'y' || choice == 'Y')
            printf("Node %d => Colour %d\n", i, colouring[i]);
    }
    cout << endl;
    printf("\nNumber of colours used (chromatic number) ==> %d\n", chromaticNumber);
    printf("Time Taken (Parallel) = %f ms\n", device_time_taken);

    if (argc == 3) {
        freopen(argv[2], "w", stdout);
        for (int i = 0; i < nodeCount; i++)
            cout << colouring[i] << " ";
        cout << endl;
    }

    // Check for correctness
    int count = 0;
    for(int u = 0; u < nodeCount; u++){
        for(int j = adjacencyListPointers[u]; j < adjacencyListPointers[u+1]; j++){
            int v = adjacencyList[j];
            if(colouring[u] == colouring[v] && u <= v ){
                if(count <= 10)
                    cout << "Conflicting colour found at nodes " << u << " and " << v << endl;
                count++;
            }
        }
    }
    cout << "Found total " << count << " conflicts!" << endl;

    // Free all memory
    delete[] colouring, adjacencyList, adjacencyListPointers, edgeListX, edgeListY;
    catchCudaError(cudaFree(device_adjacencyList), "Free");
    catchCudaError(cudaFree(device_adjacencyListPointers), "Free");
    catchCudaError(cudaFree(device_edgeListX), "Free");
    catchCudaError(cudaFree(device_edgeListY), "Free");
}
