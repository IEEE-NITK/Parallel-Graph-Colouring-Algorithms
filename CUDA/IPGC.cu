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
    int &maxDegree, int **adjacencyList, int **adjacencyListPointers) {

    int u, v;
    cin >> nodeCount >> edgeCount;

    // Use vector of vectors temporarily to input graph
    vector<int> *adj = new vector<int>[nodeCount];
    for (int i = 0; i < edgeCount; i++) {
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
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


__global__ void assignColoursKernel(int nodeCount, int *adjacencyList, int *adjacencyListPointers,
    int *device_colours, bool *device_conflicts, int maxDegree) {

    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= nodeCount || !device_conflicts[node])
        return;

    int maxColours = maxDegree + 1;
    // Create forbidden array of size maxDegree
    int *forbidden = new int[CEIL(maxColours + 1, 32)];
    if(forbidden == NULL)  {
        printf("Cuda Memory Full\n");
        return;
    }
    memset(forbidden, 0, sizeof(int) * CEIL(maxColours + 1, 32));

    for (int i = adjacencyListPointers[node]; i < adjacencyListPointers[node + 1]; i++) {
        int neighbour = adjacencyList[i];
        int ind = device_colours[neighbour] % 32;
        forbidden[device_colours[neighbour] / 32] |= (1<<ind);
    }

    for (int colour = 1; colour <= maxColours; ++colour) {
        int ind = colour % 32;
        if ((forbidden[colour / 32] & (1<<ind)) == 0) {
            device_colours[node] = colour;
            break;
        }
    }

    delete[] forbidden;
}

void assignColours(int nodeCount, int *adjacencyList, int *adjacencyListPointers, 
    int *device_colours, bool *device_conflicts, int maxDegree){

    // Launch assignColoursKernel with nodeCount number of threads
    assignColoursKernel<<<CEIL(nodeCount, MAX_THREADS), MAX_THREADS>>>(nodeCount, adjacencyList, adjacencyListPointers, 
        device_colours, device_conflicts, maxDegree);
    cudaDeviceSynchronize();
}

__global__ void detectConflictsKernel(int nodeCount, int *adjacencyList, int *adjacencyListPointers,
    int *device_colours, bool *device_conflicts, bool *device_conflictExists) {

    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= nodeCount)
        return;

    device_conflicts[node] = false;

    for (int i = adjacencyListPointers[node]; i < adjacencyListPointers[node + 1]; i++){
        int neighbour = adjacencyList[i];
        if (device_colours[neighbour] == device_colours[node] && neighbour < node) {
            //conflict
            device_conflicts[node] = true;
            *device_conflictExists = true;
        }
    }
}

bool detectConflicts(int nodeCount, int *adjacencyList, int *adjacencyListPointers,
    int *device_colours, bool *device_conflicts) {

    bool *device_conflictExists;
    bool conflictExists = false;

    catchCudaError(cudaMalloc((void **)&device_conflictExists, sizeof(bool)), "Malloc1");
    catchCudaError(cudaMemcpy(device_conflictExists, &conflictExists, sizeof(bool), cudaMemcpyHostToDevice), "Memcpy7");

    //Launch detectConflictsKernel with nodeCount number of threads
    detectConflictsKernel<<<CEIL(nodeCount, MAX_THREADS), MAX_THREADS>>>(nodeCount, adjacencyList, adjacencyListPointers, device_colours, device_conflicts, device_conflictExists);
    cudaDeviceSynchronize();

    // Copy device_conflictExists to conflictExists and return
    catchCudaError(cudaMemcpy(&conflictExists, device_conflictExists, sizeof(bool), cudaMemcpyDeviceToHost), "Memcpy6");
    
    // Free device memory
    catchCudaError(cudaFree(device_conflictExists), "Free");
    
    return conflictExists;
}

int *graphColouring(int nodeCount, int *device_adjacencyList, int *device_adjacencyListPointers, 
    int maxDegree) {

    // Boolean array for conflicts
    bool *host_conflicts = new bool[nodeCount];
    int *host_colours = new int[nodeCount];
    int *device_colours;
    bool *device_conflicts;

    // Initialize all nodes to invalid colour (0)
    memset(host_colours, 0, sizeof(int) * nodeCount);
    // Initialize all nodes into conflict
    memset(host_conflicts, true, sizeof(bool) * nodeCount);

    catchCudaError(cudaMalloc((void **)&device_colours, sizeof(int) * nodeCount), "Malloc2");
    catchCudaError(cudaMemcpy(device_colours, host_colours, sizeof(int) * nodeCount, cudaMemcpyHostToDevice), "Memcpy1");
    catchCudaError(cudaMalloc((void **)&device_conflicts, sizeof(bool) * nodeCount), "Malloc3");
    catchCudaError(cudaMemcpy(device_conflicts, host_conflicts, sizeof(bool) * nodeCount, cudaMemcpyHostToDevice), "Memcpy2");

    // Timer
    cudaEvent_t device_start, device_end;
    catchCudaError(cudaEventCreate(&device_start), "Event Create");
    catchCudaError(cudaEventCreate(&device_end), "Event Create");
    catchCudaError(cudaEventRecord(device_start), "Event Record");

    do {
        assignColours(nodeCount, device_adjacencyList, device_adjacencyListPointers, device_colours, device_conflicts, maxDegree);
    } while (detectConflicts(nodeCount, device_adjacencyList, device_adjacencyListPointers, device_colours, device_conflicts));

    // Timer
    catchCudaError(cudaEventRecord(device_end), "Event Record");
    catchCudaError(cudaEventSynchronize(device_end), "Event Synchronize");
    catchCudaError(cudaEventElapsedTime(&device_time_taken, device_start, device_end), "Elapsed time");

    // Copy colours to host and return
    catchCudaError(cudaMemcpy(host_colours, device_colours, sizeof(int) * nodeCount, cudaMemcpyDeviceToHost), "Memcpy3");

    delete[] host_conflicts;
    catchCudaError(cudaFree(device_colours), "Free");
    catchCudaError(cudaFree(device_conflicts), "Free");

    return host_colours;
}

int main(int argc, char *argv[]) {

    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <graph_input_file> [output_file]\n";
        return 0;
    }

    int nodeCount, edgeCount, maxDegree;
    int *adjacencyList = NULL, *adjacencyListPointers = NULL;
    int *device_adjacencyList, *device_adjacencyListPointers;

    char choice;
    cout << "Would you like to print the colouring of the graph? (y/n) ";
    cin >> choice;

    freopen(argv[1], "r", stdin);
    readGraph(nodeCount, edgeCount, maxDegree, &adjacencyList, &adjacencyListPointers);

    
    // Alocate device memory and copy
    catchCudaError(cudaMalloc((void **)&device_adjacencyList, sizeof(int) * (2 * edgeCount + 1)), "Malloc5");
    catchCudaError(cudaMemcpy(device_adjacencyList, adjacencyList, sizeof(int) * (2 * edgeCount + 1), cudaMemcpyHostToDevice), "Memcpy11");


    // Alocate device memory and copy
    catchCudaError(cudaMalloc((void **)&device_adjacencyListPointers, sizeof(int) * (nodeCount + 1)), "Malloc6");
    catchCudaError(cudaMemcpy(device_adjacencyListPointers, adjacencyListPointers, sizeof(int) * (nodeCount + 1), cudaMemcpyHostToDevice), "Memcpy12");

    int *colouring = graphColouring(nodeCount, device_adjacencyList, device_adjacencyListPointers, maxDegree);

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
                    cout << "Conflicting color found at nodes " << u << " and " << v << endl;
                count++;
            }
        }
    }
    cout << "Found total " << count << " conflicts!" << endl;

    // Free all memory
    delete[] colouring, adjacencyList, adjacencyListPointers;
    catchCudaError(cudaFree(device_adjacencyList), "Free");
    catchCudaError(cudaFree(device_adjacencyListPointers), "Free");
}
