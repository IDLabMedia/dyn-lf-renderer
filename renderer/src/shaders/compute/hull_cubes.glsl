#version 460

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

// -- Inputs --------------- //
layout(std430, binding = 1) readonly buffer points {
    // All the vertex points (sorted x,y,z)
    float vertices[]; // ssbo is 16 byte alighend, so vec3 will read vec4, so use manual indexing
};

/*
* get a vertex from the vertices array
*/
vec3 getVertex(uint index){
    return vec3(
    vertices[index * 3],
    vertices[index * 3 + 1],
    vertices[index * 3 + 2]
    );
}

uniform float gridSpacing; // step size of voxel grid

// -- Outputs -------------- //
layout(std430, binding = 2) buffer triangles {
    uint faces[];
};

/*
* Set a face
*/
void setFace(uint index, uint a, uint b, uint c){
    faces[index] = a;
    faces[index + 1] = b;
    faces[index + 2] = c;
}

layout(binding=3) uniform atomic_uint faceCounter;

// -- Constants ------------ //
const float EPSILON = 1e-5;
uint startVertexIdx = gl_GlobalInvocationID.x;
vec3 startVertex = getVertex(startVertexIdx);
uint verticesLen = gl_NumWorkGroups.x;


// -- Helper functions ----- //


/*
* comparator of two vertices
*/
int compareVec3(vec3 a, vec3 b) {
    if (abs(a.x - b.x) > EPSILON) return a.x < b.x ? -1 : 1;
    if (abs(a.y - b.y) > EPSILON) return a.y < b.y ? -1 : 1;
    if (abs(a.z - b.z) > EPSILON) return a.z < b.z ? -1 : 1;
    return 0;
}

/*
* binary search algorithm
*/

int binarySearch(vec3 target, uint start, uint end) {
    uint loop = 1;
    while (start < end) {
        uint mid = (start + end) / 2;
        vec3 current = getVertex(mid);

        int cmp = compareVec3(current, target);
        if (cmp == 0) {
            return int(mid); // Found
        } else if (cmp < 0) {
            start = mid + 1u;
        } else {
            end = mid;
        }
        loop++;
    }
    return -1; // Not found
}


int findCorner(uint cornerIdx){
    if(cornerIdx == 0){
        return int(startVertexIdx);
    }
    vec3 point;
    if(cornerIdx == 1){
        point = vec3(startVertex.x, startVertex.y, startVertex.z + gridSpacing);
    } else if(cornerIdx == 2){
        point = vec3(startVertex.x, startVertex.y + gridSpacing, startVertex.z);
    } else if(cornerIdx == 3){
        point = vec3(startVertex.x, startVertex.y + gridSpacing, startVertex.z + gridSpacing);
    } else if(cornerIdx == 4){
        point = vec3(startVertex.x + gridSpacing, startVertex.y, startVertex.z);
    } else if(cornerIdx == 5){
        point = vec3(startVertex.x + gridSpacing, startVertex.y, startVertex.z + gridSpacing);
    } else if(cornerIdx == 6){
        point = vec3(startVertex.x + gridSpacing, startVertex.y + gridSpacing, startVertex.z);
    } else{
        point = vec3(startVertex.x + gridSpacing, startVertex.y + gridSpacing, startVertex.z + gridSpacing);
    }

    return binarySearch(point, startVertexIdx, verticesLen);
}

/*
* Check if the corner is part of the side
*
* 0: X side : corners 0,1,2,3
* 1: Y side : corners 0,1,4,5
* 2: Z side : corners 0,2,4,6
* 3: +X side : corners 4,5,6,7
* 4: +Y side : corners 2,3,6,7
* 5: +Z side : corners 1,3,5,7
*/
bool onCubeSide(uint cornerIdx, uint sideIdx){
    if(sideIdx == 0) return cornerIdx == 0 || cornerIdx == 1 || cornerIdx == 2 || cornerIdx == 3;
    if(sideIdx == 1) return cornerIdx == 0 || cornerIdx == 1 || cornerIdx == 4 || cornerIdx == 5;
    if(sideIdx == 2) return cornerIdx == 0 || cornerIdx == 2 || cornerIdx == 4 || cornerIdx == 6;
    if(sideIdx == 3) return cornerIdx == 4 || cornerIdx == 5 || cornerIdx == 6 || cornerIdx == 7;
    if(sideIdx == 4) return cornerIdx == 2 || cornerIdx == 3 || cornerIdx == 6 || cornerIdx == 7;
    return cornerIdx == 1 || cornerIdx == 3 || cornerIdx == 5 || cornerIdx == 7;
}

/*
* Check if all corners of triangle lie on the given cube side
*/
bool triangleOnCubeSide(uint a, uint b, uint c, uint sideIdx){
    return onCubeSide(a, sideIdx) && onCubeSide(b, sideIdx) &&onCubeSide(c, sideIdx);
}

void writeTriangle(uint a, uint b, uint c){
    //uint faceID = atomicCounterIncrement(faceCounter);
    uint faceID = atomicCounterAdd(faceCounter, 3);
    setFace(faceID, a, b, c);
}

/*
* Check if ab is a diagonal of abcd
*/
bool isDiagonal(vec3 a, vec3 b, vec3 c, vec3 d){
    vec3 ab = b - a;
    vec3 ac = c - a;
    vec3 ad = d - a;

    vec3 cross_ab_ac = cross(ab, ac);
    vec3 cross_ab_ad = cross(ab, ad);

    float signC = cross_ab_ac.z;
    float signD = cross_ab_ad.z;

    return signC != signD;
}

void drawQuad(uint i, uint j, uint k, uint l){
    if(!(i < j && j < k && k < l)){
        return; // prevent quad from drawing multiple times
    }
    vec3 a = getVertex(i);
    vec3 b = getVertex(j);
    vec3 c = getVertex(k);
    vec3 d = getVertex(l);

    if(isDiagonal(a,b,c,d)){ // ab is a diagonal
        writeTriangle(i,j,k);
        writeTriangle(i,j,l);
    } else if(isDiagonal(a,c,b,d)){ // ac is a diagonal
        writeTriangle(i,k,j);
        writeTriangle(i,k,l);
    } else{ // ad is a diagonal
        writeTriangle(i,l,j);
        writeTriangle(i,l,k);
    }

}

// -- Driver code ---------- //
void main(){
    // Get a list of all the available corners
    int indices[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
    uint totalPoints = 0;
    for (uint i = 0; i < 8; ++i) {
        int idx = findCorner(i);
        if (idx >= 0) {
            indices[totalPoints] = idx;
            totalPoints++;
        }
    }

    if (totalPoints == 8){
        // Full cube so don't draw
        // however side 0,1,2
        // need to be drawn if their is no neighbour that will draw
        // those sides
        if(binarySearch(vec3(startVertex.x - gridSpacing, startVertex.y, startVertex.z), 0, verticesLen) == -1){
            drawQuad(indices[0], indices[1], indices[2], indices[3]);
        }
        if(binarySearch(vec3(startVertex.x, startVertex.y - gridSpacing, startVertex.z), 0, verticesLen) == -1){
            drawQuad(indices[0], indices[1], indices[4], indices[5]);
        }
        if(binarySearch(vec3(startVertex.x, startVertex.y, startVertex.z - gridSpacing), 0, verticesLen) == -1){
            drawQuad(indices[0], indices[2], indices[4], indices[6]);
        }
        return;
    }

    // brute-fore hull algorithm
    for(uint i = 0; i < totalPoints; ++i){
        vec3 a = getVertex(indices[i]);
        for(uint j = i + 1; j < totalPoints; ++j){
            vec3 b = getVertex(indices[j]);
            for(uint k = j + 1; k < totalPoints; ++k){
                // select any possible triangle abc
                vec3 c = getVertex(indices[k]);

                // If triangle lies on cube side 3,4 or 5
                // skip, since it will be generated by neighbouring
                // shader call
                if(triangleOnCubeSide(i,j,k,3) || triangleOnCubeSide(i,j,k,4) || triangleOnCubeSide(i,j,k,5)){
                    continue;
                }

                // Compute face normal
                vec3 normal = normalize(cross(b - a, c - a));

                // For each non triangle point, check if it lies
                // left or right or on the triangle plane.
                // if some are left and some are right, the triangle
                // is "inside" the hull, and shouldn't be generated
                bool isFace = true;
                int sideSign = 0; // indicates left or right of triangle
                int quadCorner = -1;
                for (int m = 0; m < totalPoints; ++m) {
                    if (m == i || m == j || m == k) continue; // skip triangle points
                    vec3 p = getVertex(indices[m]); // get non triangle point
                    float side = dot(normal, p - a);
                    int currentSign = (side > 0.0) ? 1 : -1; // check if point is left or right
                    if (abs(side) < EPSILON){ // point on plane, so quad
                        quadCorner = m;
                    } else if (sideSign == 0) { // sign hasn't been set yet
                        sideSign = currentSign; // set the reference direction
                    } else if (currentSign != sideSign) { // point is on other side of triangle then some previous point
                        isFace = false;
                        break;
                    }
                }

                // write the triangle to the SSBO
                if (isFace && quadCorner != -1){
                    drawQuad(indices[i], indices[j], indices[k], indices[quadCorner]);
                } else if (isFace){
                    writeTriangle(indices[i], indices[j], indices[k]);
                }
            }
        }
    }
}
