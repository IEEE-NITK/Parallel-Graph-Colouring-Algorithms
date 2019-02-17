#include <omp.h>
#include <bits/stdc++.h>
using namespace std;


/*Check if solution is correct */
void checker(vector<int> &colors, vector<vector<int> > &edges){

    for(int i=0;i<edges.size();i++){
        for(int j=0;j<edges[i].size();j++ ){
            if(colors[i]==colors[edges[i][j]]){
                printf("Conflict detected %d %d \n",i,edges[i][j]);
            }
        }
    }
}
/*Check if all nodes are colored */
bool finishColoring(vector<int> &colors,int v) {
    bool completed = true;
    int num_colored = 0;
    #pragma omp parallel for shared(completed) reduction(+:num_colored)
    for (int i = 0; i < v ; i++) {
        if (colors[i] == -1) {
            completed = false;
        }
    num_colored++;
    }
    return completed;
}
/* Check whether the neigbour colors are conflicting */
void noNeighborColors(vector<vector<int> > &edges,vector<int> &remaining_vertices, vector<int> &colors,int &nov,vector<set<int> > &noNeigh){
    #pragma omp parallel for collapse(1)
    for(int ind = 0;ind<nov;ind++){
        int vertex = remaining_vertices[ind];
        int len = edges[vertex].size();
        for(int neigh =0;neigh<len;neigh++){
            int neighbour = edges[vertex][neigh];
            if(colors[neighbour] != -1 ){
                noNeigh[vertex].insert(colors[neighbour]);
            }
        }


    }
}
/*Detect conflicts */
void detectConflicts(vector<vector<int> > &edges,vector<int> &remaining_vertices, vector<int> &colors,int &nov,int v){
    #pragma omp parallel for schedule(guided)
    for(int i=0;i<nov;i++){
        int vert = remaining_vertices[i];
         int len = edges[vert].size();
         for(int neigh =0;neigh<len;neigh++){
             int neighbor = edges[vert][neigh];
             if(colors[neighbor]==colors[vert]){
                 int minVal = min(neighbor,vert);
                 colors[minVal] = -1;
             }
         }
    }
    #pragma omp barrier
    remaining_vertices.clear();
    nov =0;
    for(int i=0;i<v;i++){
        if(colors[i]==-1){
            remaining_vertices[nov++]=i;
        }
    }

}
/* Get next possible color */
int getNextCol(int vert ,vector< vector<int> > &edges,vector<int> &colors){
    int minCol =0;
    bool valid = false;
    while(!valid){
        minCol++;
        valid = true;
        int len = edges[vert].size();
        for(int index =0;index<len && valid;index++){
            if(colors[edges[vert][index]]==minCol){
                valid = false;
            }
        }

    }
    return minCol;
}


/* Assign initial colors */
void assign_initial_color(int &nov,vector<int> &remaining_vertices,vector<int> &colors,vector< vector<int> > &edges){
    #pragma omp parallel for
    for(int ind = 0;ind < nov;ind++){
        int vert = remaining_vertices[ind];
        colors[vert] = getNextCol(vert,edges,colors);
    }
}



int main(){
    int v,e;
    cin >> v >> e;
    int nov = v;
    vector<vector<int> > edges;
    for(int i=0;i<v;i++){
        vector<int> edge;
        edges.push_back(edge);
    }
    int u,u1;
    char l;
    for(int i=0;i<e;i++){
        cin>>l>>u>>u1;
        edges[u].push_back(u1);
        edges[u1].push_back(u);

    }

    vector<int> colors;
    vector<set<int> > noNeigh;
    noNeigh.resize(nov);
    vector<int> remaining_vertices;
    for (int i = 0; i < nov; i++) {
	  remaining_vertices.push_back(i);
	}
    for(int i = 0; i < nov; i++ ){
        colors.push_back(0);
    }

	bool finish_color = false;
    while(!finish_color){
        assign_initial_color(nov,remaining_vertices,colors,edges);
        #pragma omp barrier
        detectConflicts(edges,remaining_vertices,colors,nov,v);
        #pragma omp barrier
        noNeighborColors(edges,remaining_vertices,colors,nov,noNeigh);
        #pragma omp barrier
        finish_color = finishColoring(colors,v);
        
    }
    #pragma omp barrier
    for (int i = 0; i < v; i++) {
		cout << "color " << i << ": " << colors[i] << endl;
	}
    checker(colors,edges);

}