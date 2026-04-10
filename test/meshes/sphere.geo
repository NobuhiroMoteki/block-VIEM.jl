// Unit sphere centred at the origin, meshed with linear tetrahedra.
// Used for Mie validation in later phases. The mesh size is intentionally
// coarse so .msh generation is fast in CI; finer meshes can be obtained
// with `gmsh -3 -clmax <h> sphere.geo -o out.msh`.
SetFactory("OpenCASCADE");
Sphere(1) = {0.0, 0.0, 0.0, 1.0};
Physical Volume("sphere", 1) = {1};
Mesh.CharacteristicLengthMin = 0.4;
Mesh.CharacteristicLengthMax = 0.4;
Mesh.MshFileVersion = 4.1;
