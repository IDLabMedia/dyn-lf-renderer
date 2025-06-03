#version 460

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

// -- Inputs --------------- //
layout(std430, binding = 1) readonly buffer points {
    // All the vertex points (sorted x,y,z)
    vec3 vertices[];
};

// -- Outputs -------------- //
layout(std430, binding = 2) buffer triangles {
    uvec3 faces[];
};
layout(binding=3) uniform atomic_uint faceCounter;

// -- Constants ------------ //
const float EPSILON = 1e-5;
const uint startVertexIdx = gl_GlobalInvocationID.x;
const vec3 startVertex = vertices[startVertexIdx];
const uint verticesLen = gl_NumWorkGroups.x;
const float gridSpacing = 0.01; // TODO load dyanamically


// -- Helper functions ----- //
int binarySearch(vec3 target, uint start, uint end){
    while(start < end){
        uint mid = (start + end) / 2u;
        vec3 midValue = vertices[mid];
        // Compare lexicographically
        if (abs(midValue.x - target.x) < EPSILON) {
            if (abs(midValue.y - target.y) < EPSILON) {
                if (abs(midValue.z - target.z) < EPSILON) {
                    return int(mid); // Found match
                } else if (midValue.z < target.z) {
                    start = mid + 1u;
                } else {
                    end = mid;
                }
            } else if (midValue.y < target.y) {
                start = mid + 1u;
            } else {
                end = mid;
            }
        } else if (midValue.x < target.x) {
            start = mid + 1u;
        } else {
            end = mid;
        }
    }
    return -1;
}

int findCorner(uint cornerIdx){
    if(cornerIdx == 0){
        return int(startVertexIdx);
    }
    vec3 point;
    if(cornerIdx == 1){
        point = vec3(startVertex.x + gridSpacing, startVertex.y, startVertex.z);
    } else if(cornerIdx == 2){
        point = vec3(startVertex.x + gridSpacing, startVertex.y + gridSpacing, startVertex.z);
    } else if(cornerIdx == 3){
        point = vec3(startVertex.x, startVertex.y + gridSpacing, startVertex.z);
    } else if(cornerIdx == 4){
        point = vec3(startVertex.x, startVertex.y, startVertex.z + gridSpacing);
    } else if(cornerIdx == 5){
        point = vec3(startVertex.x + gridSpacing, startVertex.y, startVertex.z + gridSpacing);
    } else if(cornerIdx == 6){
        point = vec3(startVertex.x + gridSpacing, startVertex.y + gridSpacing, startVertex.z + gridSpacing);
    } else{
        point = vec3(startVertex.x, startVertex.y + gridSpacing, startVertex.z + gridSpacing);
    }

    return binarySearch(point, startVertexIdx, verticesLen);
}

// -- Driver code ---------- //
void main(){
    // get the indices of the points in the vertex array
    int indices[8];
    uint totalPoints = 0;
    for (uint i = 0; i < 8; ++i) {
        indices[i] = findCorner(i);
        if(indices[i] >= 0){
            totalPoints++;
        }
    }
    // get a list of the actual points
    uint setIdx = 0;
    for(uint i = 0; i < 8; ++i){
        if(indices[i] >= 0){
            indices[setIdx] = indices[i];
            setIdx++;
        }
    }

    // brute-fore hull algorithm
    for(uint i = 0; i < totalPoints; ++i){
        vec3 a = vertices[i];
        for(uint j = i + 1; j < totalPoints; ++j){
            vec3 b = vertices[j];
            for(uint k = j + 1; k < totalPoints; ++k){
                // select any possible triangle abc
                vec3 c = vertices[k];

                // Compute face normal
                vec3 normal = normalize(cross(b - a, c - a));

                // For each non triangle point, check if it lies
                // left or right or on the triangle plane.
                // if some are left and some are right, the triangle
                // is "inside" the hull, and shouldn't be generated
                bool isFace = true;
                int sideSign = 0; // indicates left or right of triangle
                bool quad = false;
                for (uint m = 0; m < totalPoints; ++m) {
                    if (m == i || m == j || m == k) continue; // skip tirangle points
                    vec3 p = vertices[indices[m]]; // get non tirangle point
                    float side = dot(normal, p - a);
                    if (abs(side) < EPSILON){
                        // on the plane, so no influence, skip
                        // however, mark this triangle as part
                        // of a square surface
                        quad = true; // TODO deal with quads
                        continue;
                    }
                    int currentSign = (side > 0.0) ? 1 : -1; // check if point is left or right
                    if (sideSign == 0) { // sign hasn't been set yet
                        sideSign = currentSign; // set the reference direction
                    } else if (currentSign != sideSign) { // point is on other side of triangle then some previous point
                        isFace = false;
                        break;
                    }
                }

                // write the triangle to the SSBO
                if (isFace) {
                    uint faceID = atomicCounterIncrement(faceCounter);
                    faces[faceID] = uvec3(indices[i], indices[j], indices[k]);
                }
            }
        }
    }
}