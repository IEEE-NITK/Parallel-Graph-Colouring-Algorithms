#include <iostream>
#include <bits/stdc++.h>
using namespace std;

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

void assignColours(int nodeCount, int *adjacencyList, int *adjacencyListPointers,
                   int *colours, bool *conflicts, int maxDegree)
{

    int maxColours = maxDegree + 1;

    for (int node = 0; node < nodeCount; ++node)
    {
        if (!conflicts[node])
            break;

        bool *forbidden = new bool[maxColours + 1];
        memset(forbidden, false, sizeof(bool) * (maxColours + 1));

        for (int i = adjacencyListPointers[node]; i < adjacencyListPointers[node + 1]; i++)
        {
            int neighbour = adjacencyList[i];
            forbidden[colours[neighbour]] = true;
        }

        for (int colour = 1; colour <= maxColours; ++colour)
        {
            if (forbidden[colour] == false)
            {
                colours[node] = colour;
                break;
            }
        }

        delete[] forbidden;
    }
}

bool detectConflicts(int nodeCount, int *adjacencyList, int *adjacencyListPointers, int *colours, bool *conflicts)
{

    bool conflictExists = false;

    for (int node = 0; node < nodeCount; ++node)
    {
        if (node >= nodeCount)
            break;

        conflicts[node] = false;

        for (int i = adjacencyListPointers[node]; i < adjacencyListPointers[node + 1]; i++)
        {
            int neighbour = adjacencyList[i];

            if (colours[neighbour] == colours[node] && neighbour < node)
            {
                conflicts[node] = true;
                conflictExists = true;
            }
        }
    }

    return conflictExists;
}

int *graphColouring(int nodeCount, int *adjacencyList, int *adjacencyListPointers, int maxDegree)
{

    // Boolean array for conflicts
    bool *conflicts = new bool[nodeCount];
    int *colours = new int[nodeCount];

    // Initialize all nodes to invalid colour (0)
    memset(colours, 0, sizeof(int) * nodeCount);
    // Initialize all nodes into conflict
    memset(conflicts, true, sizeof(bool) * nodeCount);

    do
    {
        assignColours(nodeCount, adjacencyList, adjacencyListPointers, colours, conflicts, maxDegree);

    } while (detectConflicts(nodeCount, adjacencyList, adjacencyListPointers, colours, conflicts));

    delete[] conflicts;

    return colours;
}

int main(int argc, char *argv[])
{

    if (argc < 2)
    {
        cout << "Usage: " << argv[0] << " <graph_input_file> [output_file]\n";
        return 0;
    }

    char choice;
    cout << "Would you like to print the colouring of the graph? (y/n) ";
    cin >> choice;

    freopen(argv[1], "r", stdin);

    int nodeCount, edgeCount, maxDegree;
    int *adjacencyList = NULL, *adjacencyListPointers = NULL;
    int *device_adjacencyList, *device_adjacencyListPointers;

    readGraph(nodeCount, edgeCount, maxDegree, &adjacencyList, &adjacencyListPointers);
    
    clock_t start, end;
    start = clock();

    int *colouring = graphColouring(nodeCount, adjacencyList, adjacencyListPointers, maxDegree);

    end = clock();
    float time_taken = 1000.0* (end - start)/CLOCKS_PER_SEC;

    int totalColours = INT_MIN;
    for (int i = 0; i < nodeCount; i++)
    {
        totalColours = max(totalColours, colouring[i]);
        if(choice == 'y' || choice == 'Y')
            printf("Node %d => Colour %d\n", i, colouring[i]);
    }
    cout << endl;
    printf("\nNumber of colours required (chromatic number) ==> %d\n", totalColours);
    printf("Time Taken (Serial) = %f ms\n", time_taken);

    if (argc == 3)
    {
        freopen(argv[2], "w", stdout);
        for (int i = 0; i < nodeCount; i++)
            cout << colouring[i] << " ";
        cout << endl;
    }

    // Free all memory
    delete[] colouring;
}
