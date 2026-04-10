// Unit cube [0,1]^3, meshed with linear tetrahedra.
// Used for topology smoke tests and AIM grid alignment in later phases.
SetFactory("OpenCASCADE");
Box(1) = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
Physical Volume("cube", 42) = {1};
Mesh.CharacteristicLengthMin = 0.5;
Mesh.CharacteristicLengthMax = 0.5;
Mesh.MshFileVersion = 4.1;
