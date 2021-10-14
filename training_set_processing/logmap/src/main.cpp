#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/surface/vector_heat_method.h"

#include "geometrycentral/surface/direction_fields.h"

#include "args/args.hxx"
#include "imgui.h"
#include <iostream>
#include <fstream>
using namespace std;
using namespace geometrycentral;
using namespace geometrycentral::surface;

// == Geometry-central data
std::unique_ptr<ManifoldSurfaceMesh> mesh;
std::unique_ptr<VertexPositionGeometry> geometry;
float tCoef = 1.0;
std::unique_ptr<VectorHeatMethodSolver> solver;


struct SourceVert {
  Vertex vertex;
  float scalarVal = 1.;
  float vectorMag = 1.;
  float vectorAngleRad = 0.;
};
std::vector<SourceVert> sourcePoints;


void addVertexSource(size_t ind) {
  Vertex v = mesh->vertex(ind);

  // Make sure not already used
  for (SourceVert& s : sourcePoints) {
    if (s.vertex == v) {
      std::stringstream ss;
      ss << "Vertex " << v;
      std::string vStr = ss.str();
      //polyscope::warning("Vertex " + vStr + " is already a source");
      return;
    }
  }
  SourceVert newV;
  newV.vertex = v;
  sourcePoints.push_back(newV);
  //updateSourceSetViz();
}



// Some algorithm parameters
float param1 = 42.0;

// Example computation function -- this one computes and registers a scalar
// quantity
// void doWork() {
//   polyscope::warning("Computing Gaussian curvature.\nalso, parameter value = " +
//                      std::to_string(param1));
//
//   geometry->requireVertexGaussianCurvatures();
//   psMesh->addVertexScalarQuantity("curvature",
//                                   geometry->vertexGaussianCurvatures,
//                                   polyscope::DataType::SYMMETRIC);
// }

// A user-defined callback, for creating control panels (etc)
// Use ImGUI commands to build whatever you want here, see
// https://github.com/ocornut/imgui/blob/master/imgui.h
// void myCallback() {
//
//   if (ImGui::Button("do work")) {
//     doWork();
//   }
//
//   ImGui::SliderFloat("param", &param1, 0., 100.);
// }

int main(int argc, char **argv) {

  // Configure the argument parser
  args::ArgumentParser parser("geometry-central & Polyscope example project");
  args::Positional<std::string> inputFilename(parser, "mesh", "A mesh file.");
  args::Positional<std::string> inputChosenVerticesFilename(parser, "chosen_vertices", "A mesh file.");
  args::Positional<std::string> outputFilename(parser, "output filename", "A mesh file.");
  args::Positional<std::string> NNFilename(parser, "nearest neighbors in chosen vertices", "A mesh file.");

  // Parse args
  try {
    parser.ParseCLI(argc, argv);
  } catch (args::Help) {
    std::cout << parser;
    return 0;
  } catch (args::ParseError e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return 1;
  }
  // Make sure a mesh name was given
  if (!inputFilename) {
    std::cerr << "Please specify a mesh file as argument" << std::endl;
    return EXIT_FAILURE;
  }

  //std::cout <<"****************"<< args::get(inputFilename)<< args::get(inputChosenVerticesFilename) << '\n';

  // load vector of chosen Vertices

  vector<int> chosen_vertices;
  std::fstream chosen_vertices_file(args::get(inputChosenVerticesFilename), std::ios_base::in);
  int a;
  while (chosen_vertices_file >> a)
  {
    chosen_vertices.push_back(a);
    //printf("%d ", a);
  }

  vector<int> NN;
  std::cout <<"c++ logmap code running "<< NNFilename << '\n';
  std::fstream NN_vertices_file(args::get(NNFilename), std::ios_base::in);
  while (NN_vertices_file >> a)
  {
    NN.push_back(a);
    //printf("%d ", a);
  }




  // Initialize polyscope
  //polyscope::init();

  // Set the callback function
  //polyscope::state::userCallback = myCallback;

  // Load mesh
  std::tie(mesh, geometry) = readManifoldSurfaceMesh(args::get(inputFilename));

  // log map
  if (solver == nullptr) {
    solver.reset(new VectorHeatMethodSolver(*geometry, tCoef));
  }
  geometry->requireVertexIndices();
  addVertexSource(0);
  Vertex sourceV = sourcePoints[0].vertex;

  ofstream result_file;
  result_file.open(args::get(outputFilename));
  // for(Vertex source : mesh->vertices()) {
  //   VertexData<Vector2> logmap = solver->computeLogMap(source);
  //   int n = mesh->nVertices();
  //   std::cout << "******************" << '\n';
  //   for(int v = 0; v<n ; v++){
  //     result_file << logmap[v][0]<<" "<<logmap[v][1]<<"\n";
  //   }
  // }
  int counter = 0;
  int n_neighbors = NN.size()/chosen_vertices.size();
  std::cout << "n neighbors" << n_neighbors<< '\n';
  for ( auto i = chosen_vertices.begin(); i != chosen_vertices.end(); i++ ) {
    Vertex source = mesh->vertex((size_t) *i) ;
    VertexData<Vector2> logmap = solver->computeLogMap(source);
    int n = mesh->nVertices();
    std::cout << counter <<" "<< *i << '\n';
    //for(int v = 0; v<n ; v++){
    // for ( auto v= chosen_vertices.begin(); v != chosen_vertices.end(); v++ ) {
    //   result_file << *v<<" " <<logmap[*v][0]<<" "<<logmap[*v][1]<<"\n";
    // }
    for (int k = counter*n_neighbors; k < (counter+1)*n_neighbors; ++k){
      //std::cout << NN.size() << '\n';
      result_file << NN[k]<<" " <<logmap[NN[k]][0]<<" "<<logmap[NN[k]][1]<<"\n";
      //result_file << logmap[NN[k]][0]<<"\n";
    }
    counter +=1;


  }

  result_file.close();

  return EXIT_SUCCESS;
}
