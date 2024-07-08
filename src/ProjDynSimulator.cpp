/*
********************************************************************************
MIT License

Copyright(c) 2018 Christopher Brandt
Copyright (c) 2024 Shaimaa Monem

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
********************************************************************************
*/
#include <iostream>
#include <fstream>
#include <omp.h>
#include <limits>

#include "ProjDynSimulator.h"
#include "ProjDynConstraints.h"
#include "ProjDynUtil.h"
#include "ProjDynMeshSampler.h"
#include "StopWatch.h"
#include "ProjDynTetGen.h"
#include "ProjDynRHSInterpol.h"

#include <igl/readOFF.h>
#include <igl/writeOFF.h>     // to store meshes in .off format
#include <igl/dqs.h>
#include "Eigen/Core"
#include <unsupported/Eigen/MatrixFunctions>

using namespace PD;


ProjDynSimulator::ProjDynSimulator
(PDTriangles& triangles, PDPositions& initialPositions, PDPositions& initialVelocities,
	PDScalar timeStep,
	int numPosPODModes,
	std::string pca_directory,
	int numSPLOCSModes,
	std::string splocs_directory,
	int numSamplesPosSubspace,
	PDScalar baseFunctionRadius,
	int interpolBaseSize,
	PDScalar rhsInterpolWeightRadius,
	int numConstraintSamples,
	PDScalar massPerUnitArea,     
	PDScalar dampingAlpha,
	bool makeTets,
	std::string meshURL,
	PDScalar rhsRegularizationWeight,
	PDScalar yTrans) :

	m_flatBending(false),
	m_rhsRegularizationWeight(rhsRegularizationWeight),

	m_parallelVUpdate(false),
	m_parallelVUpdateBSize(PROJ_DYN_VPOS_BLOCK_SIZE),

	m_meshURL(meshURL),
	m_meshName(PD::getMeshName(meshURL)),
	m_numTets(0),
	m_hasTetrahedrons(false),
	m_rayleighDampingAlpha(dampingAlpha),

	// LBS bases for position parameters:
	m_usingSkinSubspaces(numSamplesPosSubspace > 0),
	m_rhsInterpolation(numConstraintSamples > 0),
	m_baseFunctionRadius(baseFunctionRadius),

	// LBS bases for constraints parameters:
	m_rhsInterpolBaseSize(interpolBaseSize),
	m_numSamplesPosSubspace(numSamplesPosSubspace),
	m_numConstraintSamples(numConstraintSamples),

	// Snapshots bases for position parameters:
	m_numPosPODModes(numPosPODModes),
	m_usingPODPosSubspaces(numPosPODModes > 0), 
	
	isPODLocal(true),
	isLocalPOD_Sparse(true),    
	isPODBasisOrthogonal(PCA_POSITION_ORTHOGONAL == "_Orthogonalized"),
	m_numPosSPLOCSModes(numSPLOCSModes),
	m_usingSPLOCSPosSubspaces(numSPLOCSModes >0),       // when the basis being loaded as POD basis are acu]tually SPLOCS basis, then specify here!
	podUsedVerticesOnly(numConstraintSamples > 0),
	
	m_usePosSnapBases(numPosPODModes > 0 || numSPLOCSModes > 0),
	//TODO: DEIM part-----------------------------
	m_numQDEIMModes(0),
	m_usingQDEIMComponents(false),
	m_TetStrainOnly(true),
	
	m_solveDeimLS(false),   // false: use deim for modes of blocks, interpolationBlocks are x/y/z separate
							// true: stack interpolationBlocks from x/y/z and use LS to find the reduced constraints projections
	
	// Full order simulations (where snapshots might be collected)
	m_usingPosSubspaces(numPosPODModes > 0 || numSPLOCSModes > 0 || numSamplesPosSubspace > 0),
	recordingSTpSnapshots(false), 
	recordingPSnapshots(false),
	// note recording happens only for one constraint at a time
	recordingTetStrainOnly(false),
	
	// Time measures and solver parameters:
	m_localStepStopWatch(10000, 10000),
	m_globalStepStopWatch(10000, 10000),
	m_totalStopWatch(10000, 10000),
	m_localStepOnlyProjectStopWatch(10000, 10000),
	m_localStepRestStopWatch(10000, 10000),
	m_surroundingBlockStopWatch(10000, 10000),
	m_updatingVPosStopWatch(10000, 10000),
	m_fullUpdateStopWatch(10000, 10000),
	m_precomputationStopWatch(10, 10),

	m_numIterations(-1),
	m_floorCoordinate(-1),
	m_recomputeFactorization(false),
	m_floorCollisionWeight(0),

	m_constraintSamplesChanged(true),
	m_numRefactorizations(0),
	m_constraintSummationStopWatch(10000, 10000),
	m_momentumStopWatch(10000, 10000),
	m_multiplicationForPosUpdate(10000, 10000),
	m_sortingForPosUpdate(10000, 10000),
	m_timeStep(timeStep),

	m_tetrahedrons(0, 4),
	m_frameCount(0),

	m_usedVertexMap(nullptr),

	m_rhsInterpolReusableWeights(0, 0),
	m_rhsInterpolWeightRadiusMultiplier(rhsInterpolWeightRadius),

	m_useSparseMatricesForSubspace(PROJ_DYN_SPARSIFY)

#ifdef PROJ_DYN_USE_CUBLAS
	,
	m_usedVertexUpdater(nullptr),
	m_rhsEvaluator(nullptr),
	m_usedVertexUpdaterSparse(nullptr)
#endif
{

	m_PCABasesDir = pca_directory;
	m_SPLOCSBasesDir = splocs_directory;

	once = false;

	// Construct position and triangle matrices
	m_numOuterVertices = initialPositions.rows();
	m_positions = initialPositions;
	for (unsigned int v = 0; v < m_positions.rows(); v++) m_positions(v, 1) += yTrans;
	m_numTriangles = triangles.rows();
	m_triangles = triangles;


	// Create tet mesh out of triangle mesh, if desired
	// The new vertices and triangles are appended to the old matrices
	// m_numVertices will be the number of all vertices, m_numOuterVertices
	// is the number of only the original vertices
	if (makeTets) {
		TetGen test;
		PDPositions positionsNew;
		if (test.tetrahedralizeMesh(m_positions, m_triangles, positionsNew, m_tetrahedrons)) {
			m_positions = positionsNew;
			m_hasTetrahedrons = true;
			m_numTets = m_tetrahedrons.rows();
		}
		else {
			std::cout << "Error: Could not generate tets! Mesh has self-intersections or other problems..." << std::endl;
		}
	}

	m_numVertices = m_positions.rows();
	std::cout << "Initiating vertex sampler (i.e. distance fields) ..." << std::endl;
	
	if(m_rhsInterpolation || m_usingSkinSubspaces){
	m_sampler.init(m_positions, m_triangles, m_tetrahedrons);
	}

	std::cout << "Initiatlizing external force and velocity vectors ..." << std::endl;

	m_frictionCoeff = 0;

	m_fExt.resize(m_numVertices, 3);
	m_fExt.setZero();

	m_fGravity.resize(m_numVertices, 3);
	m_fGravity.setZero();
	m_velocities = initialVelocities;  // given to the ProjDynSimulator 

	// compute vertex masses
	if (!m_hasTetrahedrons) {
		m_vertexMasses = vertexMasses(m_triangles, m_positions);   
	}
	else {
		m_vertexMasses = vertexMasses(m_tetrahedrons, m_positions);
	}

	std::cout << "Average vertex mass: " << m_vertexMasses.sum() / (float)m_numVertices << std::endl;
	m_totalMass = m_vertexMasses.sum();

	// Normalize vertex masses to integrate to 1 for numerical stability
	m_normalization = (1. / m_totalMass);
	m_vertexMasses *= m_normalization * massPerUnitArea;  // massPerUnitArea just given fixed = 2!
	std::cout << "Average vertex mass after normalization : " << m_vertexMasses.sum() / (float)m_numVertices << std::endl;
	
	
	std::vector<Eigen::Triplet<PDScalar>> massEntries;
	massEntries.reserve(m_numVertices);
	for (int v = 0; v < m_numVertices; v++) {
		massEntries.push_back(Eigen::Triplet<PDScalar>(v, v, m_vertexMasses(v)));   // m_vertexMasses is a (N,) vector
	}
	m_massMatrix = PDSparseMatrix(m_numVertices, m_numVertices);
	m_massMatrix.setFromTriplets(massEntries.begin(), massEntries.end());   // m_massMatrix isDiagonal (N, N) matrix
	
	
	
	std::vector<Eigen::Triplet<PDScalar>> massInvEntries;
	massInvEntries.reserve(m_numVertices);
	for (int v = 0; v < m_numVertices; v++) {
		massInvEntries.push_back(Eigen::Triplet<PDScalar>(v, v, 1./m_vertexMasses(v)));   // m_vertexMasses is a (N,) vector
	}
	
	m_massMatrixInv.resize(m_numVertices, m_numVertices);    // inverse of m_massMatrix
	m_massMatrixInv.setFromTriplets(massInvEntries.begin(), massInvEntries.end());   
	
	
	/*
	// uncomment to store the mass matrix for a specific mesh
	// note: a .bin file is required to compute the POD basis python code.
	PDMatrix funcs ;
	funcs.setZero(m_numVertices, 1);
	for (int v = 0; v < m_numVertices; v++) {
		funcs(v,0) =m_vertexMasses(v);   
	}
	// store massMartix in a binary
	PD::storeBaseBinary(funcs, PD::getMeshFileName(m_meshURL, "_massMatrix.bin"));
	*/
	
	// Print some stat about the mesh:
	m_vertexStars = makeVertexStars(m_numVertices, m_numTriangles, m_triangles);   // vector of edges, at each vert gives a vector listing al edges meeting the vert
	int vStarSize = 0;
	for (int v = 0; v < m_numVertices; v++) {
		vStarSize += m_vertexStars.at(v).size();
	}
	std::cout << "Average vertex star size: " << ((float)vStarSize / (float)m_numVertices) << std::endl;

	for (unsigned int i = 0; i < m_numVertices; i++) {
		m_allVerts.push_back(i);                       // vector of unsigned intergers contains the inices of all vertices!
	}

	m_isSetup = false;      

	m_rhs.setZero(m_numVertices, 3);                      // initialize momentum term for the RHS

	omp_set_num_threads(PROJ_DYN_NUM_THREADS);
#ifndef EIGEN_DONT_PARALLELIZE       // is set beggining dobubleHRPD
	Eigen::setNbThreads(PROJ_DYN_EIGEN_NUM_THREADS);
	Eigen::initParallel();
#endif

	// create directories to store position snapshots
	// note snapshots are stored only if STORE_FRAMES_OFF is sat true
	m_meshSnapshotsDirectory = "../../../results/";
	if (CreateDirectory(m_meshSnapshotsDirectory.c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError())
	{
		m_meshSnapshotsDirectory = m_meshSnapshotsDirectory + m_meshName + "/";
		if (CreateDirectory(m_meshSnapshotsDirectory.c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError())
		{
			m_meshSnapshotsDirectory = m_meshSnapshotsDirectory +  "_gravitationalFall/"; 
			if (CreateDirectory(m_meshSnapshotsDirectory.c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError())
			{
				m_meshSnapshotsDirectory = m_meshSnapshotsDirectory + "position_snapshots/";
				if (CreateDirectory(m_meshSnapshotsDirectory.c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError())
				{
					std::cout << "Snapshots directory created!: " << m_meshSnapshotsDirectory << std::endl;
				}
			}
		}
	}
}

void ProjDynSimulator::finalizeBaseFunctions() {
	m_baseFunctionsTransposed = m_baseFunctions.transpose();
	//m_baseFunctionsSquared = m_baseFunctionsTransposed * m_baseFunctions;
}

void ProjDynSimulator::finalizePODBaseFunctions() {
	m_baseXFunctionsTransposed = m_baseXFunctions.transpose();
	m_baseYFunctionsTransposed = m_baseYFunctions.transpose();
	m_baseZFunctionsTransposed = m_baseZFunctions.transpose();	
}


PDPositions& ProjDynSimulator::getPositions()   // for simulation rendering in doubleHRPD (main.cpp)
{
	
	return m_positions;
}

/*
PDPositions& ProjDynSimulator::getVelocities()   // can be for simulation rendering/interaction in doubleHRPD (main.cpp)
{
	return m_velocities;
}
*/
void PD::ProjDynSimulator::recomputeWeightedForces() {
	m_fExtWeighted.resize(m_numVertices, 3);
	m_fGravWeighted.resize(m_numVertices, 3);
	
	//m_fExt can be set using setExternalForces(PDPositions fExt) in main.cpp , otherwise zeros.
	// m_fGravity is set using addGravity(PDScalar gravity) in main.cpp, otherwise zeros.
	
	PROJ_DYN_PARALLEL_FOR
			for (int v = 0; v < m_numVertices; v++) {
				for (int d = 0; d < 3; d++) {
					// the smaller the m_vertexMasses(v) the bigger the weighted force
					m_fExtWeighted(v, d) = m_fExt(v, d) * (1. / m_vertexMasses(v));     
					m_fGravWeighted(v, d) = m_fGravity(v, d) * (1. / m_vertexMasses(v));  
				}
			}
	// External and gravitational forces*(h^2))
	m_fExtWeighted *= m_timeStep * m_timeStep;
	m_fGravWeighted *= m_timeStep * m_timeStep;
		
	if (m_usingSkinSubspaces && m_isSetup && !m_usePosSnapBases) {

		projectToSubspace(m_fExtWeightedSubspace, m_fExtWeighted, false);
		projectToSubspace(m_fGravWeightedSubspace, m_fGravWeighted, false);
	}
	if (m_usePosSnapBases && m_isSetup && !m_usingSkinSubspaces) {
		
		m_fExtWeightedSubspace.resize(m_baseXFunctions.cols(), 3);
		m_fGravWeightedSubspace.resize(m_baseXFunctions.cols(), 3);
	
	
		projectToPODSubspace(m_fExtWeightedSubspace, m_fExtWeighted, false);
		projectToPODSubspace(m_fGravWeightedSubspace, m_fGravWeighted, false);
	}
	
	if (m_fExtWeightedSubspace.hasNaN() || m_fGravWeightedSubspace.hasNaN()) {
		std::cout << "Error: NaN entries in recomputed external forces" << std::endl;
	}
}

void PD::ProjDynSimulator::projectToSubspace(PDPositions& b, PDPositions& x, bool isBasisOrthogonal)
{
	// projecting to position skinning subspace using U.transpose
	if(!isBasisOrthogonal){
		// solves normal equation: 
		// (m_baseFunctionsTransposed * m_massMatrix * m_baseFunctions) b = m_baseFunctionsTransposed * m_massMatrix * x
		// for b: a reduces pos/ velo/..
		PDPositions rhs = m_baseFunctionsTransposed * m_massMatrix * x;
		b.resize(m_baseFunctionsTransposed.rows(), 3);
		for (int d = 0; d < 3; d++) {
			b.col(d) = m_subspaceSolver.solve(rhs.col(d));  
		}
	}
	else{
		PDMatrix projectionMat = m_baseFunctionsTransposed * m_massMatrix;
		
		PROJ_DYN_PARALLEL_FOR
			for (int d = 0; d < 3; d++) {
				b.col(d) = projectionMat * x(d);
			}
	}
}


void PD::ProjDynSimulator::projectToPODSubspace(PDPositions& subPos, PDPositions& fullPos, bool isBasisOrthogonal)
{
	if(!isBasisOrthogonal){
		
		PDPositions rhs;
		rhs.setZero(m_baseXFunctionsTransposed.rows() , 3);
		rhs.col(0) = m_baseXFunctionsTransposed * m_massMatrix * fullPos.col(0);
		rhs.col(1) = m_baseYFunctionsTransposed * m_massMatrix * fullPos.col(1);
		rhs.col(2) = m_baseZFunctionsTransposed * m_massMatrix * fullPos.col(2);
		subPos.resize(m_baseXFunctionsTransposed.rows(), 3);
		
		// projecting in parallel 
		#pragma omp parallel
		#pragma omp single nowait
		{
		#pragma omp task
		    subPos.col(0) = m_subspaceXSolver.solve(rhs.col(0));
		#pragma omp task
		    subPos.col(1) = m_subspaceYSolver.solve(rhs.col(1));
		#pragma omp task
		    subPos.col(2) = m_subspaceZSolver.solve(rhs.col(2));
		}
		
		if(m_subspaceXSolver.info()!= Eigen::Success || m_subspaceYSolver.info()!= Eigen::Success  || m_subspaceZSolver.info()!= Eigen::Success  ) {
		  // solving not sucssesful
		  std::cout << "FATAL ERROR! projection to nonOrthogonal basis failed" << std::endl;
		  return;
		}
		
	}
	else{    
		//TODO: orthogonal case not yet working	
		PDMatrix projectionXMat = m_baseXFunctionsTransposed;
		PDMatrix projectionYMat = m_baseYFunctionsTransposed; 
		PDMatrix projectionZMat = m_baseZFunctionsTransposed; 
		
		 subPos.col(0) = projectionXMat * m_massMatrix * fullPos.col(0); 
		 subPos.col(1) = projectionYMat * m_massMatrix * fullPos.col(1);
		 subPos.col(2) = projectionZMat * m_massMatrix * fullPos.col(2);
	}
}

void PD::ProjDynSimulator::projectToSparsePODSubspace(PDPositions& subPos, PDPositions& fullPos, bool isBasisOrthogonal)
{
	if(!isBasisOrthogonal){
		
		PDPositions rhs;
		rhs.setZero(m_baseXFunctionsTransposed.rows() , 3);
		rhs.col(0) = m_baseXFunctionsTransposedSparse * m_massMatrix * fullPos.col(0);
		rhs.col(1) = m_baseYFunctionsTransposedSparse * m_massMatrix * fullPos.col(1);
		rhs.col(2) = m_baseZFunctionsTransposedSparse * m_massMatrix * fullPos.col(2);
		subPos.resize(m_baseXFunctionsTransposedSparse.rows(), 3);
		
		// Projecting sparse in parallel
		#pragma omp parallel
		#pragma omp single nowait
		{
		#pragma omp task
		    subPos.col(0) = m_subspaceXSparseSolver.solve(rhs.col(0));
		#pragma omp task
		    subPos.col(1) = m_subspaceYSparseSolver.solve(rhs.col(1));
		#pragma omp task
		    subPos.col(2) = m_subspaceZSparseSolver.solve(rhs.col(2));
		} 
		
		if(m_subspaceXSolver.info()!= Eigen::Success || m_subspaceYSolver.info()!= Eigen::Success  || m_subspaceZSolver.info()!= Eigen::Success  ) {
		  // solving not sucssesful
		  std::cout << "FATAL ERROR! projection to nonOrthogonal basis failed" << std::endl;
		  return;
		}
		
	}
	else{  	
		PDMatrix projectionXMat = m_baseXFunctionsTransposedSparse* m_massMatrix ; // * (1./m_massMatrix.sum());
		PDMatrix projectionYMat = m_baseYFunctionsTransposedSparse* m_massMatrix ; // * (1./m_massMatrix.sum());
		PDMatrix projectionZMat = m_baseZFunctionsTransposedSparse* m_massMatrix ; // * (1./m_massMatrix.sum());
		
		 subPos.col(0) = projectionXMat * fullPos.col(0); 
		 subPos.col(1) = projectionYMat * fullPos.col(1);
		 subPos.col(2) = projectionZMat * fullPos.col(2);
			 
		/*
		#pragma omp parallel
		#pragma omp single nowait
		{
		#pragma omp task

		    subPos.col(0) = projectionXMat * fullPos.col(0);  // if you remove .col() you can surprisingly see somthing!
		#pragma omp task
		    subPos.col(1) = projectionYMat * fullPos.col(1);
		#pragma omp task
		    subPos.col(2) = projectionZMat * fullPos.col(2);
		} */
		

	}
}



/* collect smaples (vertices/handles) to create SkinSubspaces for constraints,
   Only used in case using rhsInterpolation. */
void PD::ProjDynSimulator::createConstraintSampling(unsigned int numSamples) {
	// If not using snapshots, or loading was not successful, use the semi-
	// equidistant sampling approach via the heat method
	
	if(m_usingSkinSubspaces){
		m_constraintVertexSamples = m_samples;  // (~handles verts) created at the skin posSubscape step.
	}
	else{   // if not using SkinSubspaces for positions, no samples have been sampled so far, so go getSamples
		 m_constraintVertexSamples = m_sampler.getSamples(numSamples);  // use heatMethod and furthest point method
	}
	
	
	if (m_constraintVertexSamples.empty()) {    
		m_constraintVertexSamples.push_back(std::rand() % m_numVertices);
	}
	// extendSamples : in case of a difference between numSamples and the samples in m_constraintVertexSamples
	m_sampler.extendSamples(numSamples, m_constraintVertexSamples);  // increase number of handles
	
	PD::fillSamplesRandomly(m_constraintVertexSamples, numSamples, m_numVertices - 1);   // why do we need to fill randomly?

	std::sort(m_constraintVertexSamples.begin(), m_constraintVertexSamples.end());
	m_constraintVertexSamples.erase(std::unique(m_constraintVertexSamples.begin(), m_constraintVertexSamples.end()), m_constraintVertexSamples.end());

	m_constraintTriSamples.clear();
	m_constraintTriSamples.reserve(m_constraintVertexSamples.size());
	for (unsigned int vInd : m_constraintVertexSamples) {
		if (m_vertexStars.at(vInd).size() > 0) {
			m_constraintTriSamples.push_back(m_vertexStars.at(vInd).at(0).t1);  // constrain the first tri of the first edge in the list of vInd nighboring edges
		}
	}
	std::sort(m_constraintTriSamples.begin(), m_constraintTriSamples.end());
	m_constraintTriSamples.erase(std::unique(m_constraintTriSamples.begin(), m_constraintTriSamples.end()), m_constraintTriSamples.end());

	if (m_hasTetrahedrons) {
		m_tetsPerVertex = makeTetsPerVertexList(m_numVertices, m_tetrahedrons);   // vector of vectors,at each vInd a vector of the nighboring tets

		m_constraintTetSamples.clear();
		m_constraintTetSamples.reserve(m_constraintVertexSamples.size());
		for (unsigned int vInd : m_constraintVertexSamples) {
			if (m_tetsPerVertex.at(vInd).size() > 0) {
				m_constraintTetSamples.push_back(m_tetsPerVertex.at(vInd).at(0));   // constrain only first tet in the tets list of vInd
			}
		}
		std::sort(m_constraintTetSamples.begin(), m_constraintTetSamples.end());
		m_constraintTetSamples.erase(std::unique(m_constraintTetSamples.begin(), m_constraintTetSamples.end()), m_constraintTetSamples.end());
	}
	
}

void PD::ProjDynSimulator::createQDEIMConstraintTetStrainSampling(){
	
	if(qdeimBlocks.cols() == 0){
		std::cout << "Fatal Error! QDEIM blocks indecies can not be found" << std::endl;
		return;
	}
	
	m_constraintVertexSamples.resize(qdeimBlocks.rows());
	m_constraintTriSamples.resize(qdeimBlocks.rows());
	m_constraintTetSamples.resize(qdeimBlocks.rows());
	
	// constrained tets 
	for (int b= 0; b <= qdeimBlocks.rows(); b++){
		m_constraintTetSamples[b] = qdeimBlocks(b, 0);   // sorted list of tet blocks picked by QDEIM algorithm
		//std::cout << m_constraintTetSamples[b] << std::endl;
	} 
	
	// assure order and uniquness
	std::sort(m_constraintTetSamples.begin(), m_constraintTetSamples.end());
	m_constraintTetSamples.erase(std::unique(m_constraintTetSamples.begin(), m_constraintTetSamples.end()), m_constraintTetSamples.end());
		
	// constrained verts
	m_vertsPerTet = makeVertsPerTetList(m_numVertices, m_tetrahedrons);
	m_constraintVertexSamples.clear();
	m_constraintVertexSamples.reserve(m_constraintTetSamples.size());
	
	for (unsigned int tetInd : m_constraintTetSamples) {
		if (m_vertsPerTet.at(tetInd).size() > 0) {
			m_constraintVertexSamples.push_back(m_vertsPerTet.at(tetInd).at(0));   // constrain only first vert in the verts list of tetInd
		}
	}
	
	// assure order and uniquness
	std::sort(m_constraintVertexSamples.begin(), m_constraintVertexSamples.end());
	m_constraintVertexSamples.erase(std::unique(m_constraintVertexSamples.begin(), m_constraintVertexSamples.end()), m_constraintVertexSamples.end());
	
	// constrained tris
	m_constraintTriSamples.clear();
	m_constraintQDEIMTriSamples.reserve(m_constraintVertexSamples.size());
	for (unsigned int vInd : m_constraintVertexSamples) {
		//std::cout << vInd << std::endl;
		if (m_vertexStars.at(vInd).size() > 0) {
			m_constraintTriSamples.push_back(m_vertexStars.at(vInd).at(0).t1);  // constrain the first tri of the first edge in the list of vInd nighboring edges
		}
	}
	
	std::sort(m_constraintTriSamples.begin(), m_constraintTriSamples.end());
	m_constraintTriSamples.erase(std::unique(m_constraintTriSamples.begin(), m_constraintTriSamples.end()), m_constraintTriSamples.end());
	
	
	std::cout << "Total number of sampled tets/tris/verts are: " << m_constraintTetSamples.size()<< " " << m_constraintTriSamples.size()<< " " << m_constraintVertexSamples.size() << std::endl;
}
// used only in cases using skinSubspaces or rhsInterpolation
void PD::ProjDynSimulator::evaluatePositionsAtUsedVertices(PDPositions& usedPos, PDPositions& subPos)
{
	int vSize = m_usedVertices.size();
	//std::cout<< vSize << std::endl;
	int subSize = subPos.rows();
	if(m_usingSkinSubspaces){
	int i = 0;
	PROJ_DYN_PARALLEL_FOR
		for (i = 0; i < vSize; i++) {
			int nnz = m_usedVerticesBase[i].size();
			for (int d = 0; d < 3; d++) {
				PDScalar sum = 0;
				for (int j = 0; j < nnz; j++) sum += m_usedVerticesBase[i].at(j) * subPos(m_usedVerticesBaseNNZ[i].at(j), d);
				usedPos(i, d) = sum;
			}
		}
	}
	else if(m_usePosSnapBases){
	int i = 0;
	PROJ_DYN_PARALLEL_FOR
		for (i = 0; i < vSize; i++) {
			int nnz = m_usedVerticesXBase[i].size();
		
			PDScalar sumX = 0;
			for (int j = 0; j < nnz; j++) sumX += m_usedVerticesXBase[i].at(j) * subPos(m_usedVerticesXBaseNNZ[i].at(j), 0);
			usedPos(i, 0) = sumX;
		
		}
		
	PROJ_DYN_PARALLEL_FOR
		for (i = 0; i < vSize; i++) {
			int nnz = m_usedVerticesYBase[i].size();
		
			PDScalar sumY = 0;
			for (int j = 0; j < nnz; j++) sumY += m_usedVerticesYBase[i].at(j) * subPos(m_usedVerticesYBaseNNZ[i].at(j), 1);
			usedPos(i, 1) = sumY;
		
		}
		
	PROJ_DYN_PARALLEL_FOR
		for (i = 0; i < vSize; i++) {
			int nnz = m_usedVerticesZBase[i].size();
		
			PDScalar sumZ = 0;
			for (int j = 0; j < nnz; j++) sumZ += m_usedVerticesZBase[i].at(j) * subPos(m_usedVerticesZBaseNNZ[i].at(j), 2);
			usedPos(i, 2) = sumZ;
		
		}
		
		
	}
	
	
}



// resolves collision for all cases:  it changes the matrx "posCorrect" at the desired vertex "v"
void PD::ProjDynSimulator::resolveCollision(unsigned int v, PDPositions & pos, PDPositions & posCorrect)
{
	posCorrect.row(v) = pos.row(v);  // set the correction equal to current position of v

	if (m_floorCollisionWeight > 0) {
		
		if (pos(v, 1) < m_floorHeight) {
			m_collisionCorrection = true;
			pos(v, 1) = m_floorHeight;   // change the posiotion of the the vertex to equal the fllor hight

		}
	}

	for (CollisionObject& col : m_collisionObjects) {
		PD3dVector posV = pos.row(v);
		if (col.resolveCollision(posV)) {
			m_collisionCorrection = true;
			pos.row(v) = posV.transpose();
		}
	}


	posCorrect.row(v) -= pos.row(v);  
	posCorrect.row(v) *= -1.;     // now the position correction is:  - (current posiotn - the foolr hight)
}


/* Updates the actual positions using the subspace positions, but if rhs interpolation is used
only updates positions in the list m_usedVertices, and expects that the vector fullPos is
size m_usedVertices.size() and will fill it corresponding to the list of used vertices.
Otherwise expects the usual full position vector. */
// called only when either using skinSbspaces or rhsInterpolation
// note: in this code, it was used only when we have both! it does not work correctly otherwise!! 
void PD::ProjDynSimulator::updatePositionsSampling(PDPositions& fullPos, PDPositions& subPos, bool usedVerticesOnly)
{
	if (usedVerticesOnly) {      
			
#ifdef PROJ_DYN_USE_CUBLAS
		PDScalar one = 1;
		for (int d = 0; d < 3; d++) {

			m_multiplicationForPosUpdate.startStopWatch();
			// m_usedVertexUpdater is a pointer. .data() retuns a pointer to a block of memory
			// CUDAMatrixVectorMultiplier* m_usedVertexUpdater;
			if (m_useSparseMatricesForSubspace) {
				//m_usedVertexUpdaterSparse->mult(subPos.data() + (d * subPos.rows()), fullPos.data() + (d * fullPos.rows()), one);
				/* Sparse multiplication does NOT seem worth it in this case */
				m_usedVertexUpdater->mult(subPos.data() + (d * subPos.rows()), fullPos.data() + (d * fullPos.rows()), one);
			}
			else {
				m_usedVertexUpdater->mult(subPos.data() + (d * subPos.rows()), fullPos.data() + (d * fullPos.rows()), one);
			}
			m_multiplicationForPosUpdate.stopStopWatch();
		}
#else
		evaluatePositionsAtUsedVertices(fullPos, subPos);
#endif
	}
	else {
		fullPos = m_baseFunctions * subPos;

	}
}

void PD::ProjDynSimulator::updatePODPositionsSampling(PDPositions& fullPos, PDPositions& subPos, bool usedVerticesOnly)
{
	if (!usedVerticesOnly) {     
	
		#pragma omp parallel
		#pragma omp single nowait
		{
		#pragma omp task
			fullPos.col(0) = m_baseXFunctions * subPos.col(0);
		#pragma omp task
			fullPos.col(1) = m_baseYFunctions * subPos.col(1);
		#pragma omp task
			fullPos.col(2) = m_baseZFunctions * subPos.col(2);
		} 	
	}

	else {  // usedVerticesOnly
		evaluatePositionsAtUsedVertices(fullPos, subPos);
	}
}



void PD::ProjDynSimulator::addConstraintSample(ProjDynConstraint * c)
{
	m_sampledConstraints.push_back(c);
	m_constraintSamplesChanged = true;
}

/*
void PD::ProjDynSimulator::setExamplePoses(std::vector<PDPositions> exPoses, PDScalar generalWeight, bool forSprings)
{
	std::cout << "===PD::ProjDynSimulator::setExamplePoses===" << std::endl;
	if (!m_hasTetrahedrons) {
		std::cout << "Example poses currently only implemented when using tet constraints." << std::endl;
		return;
	}
	else {
		// We need to extend the surface positions to the rest of the tet-mesh, which is
		// done by minimizing the internal forces from the tet-strain energy
		// while constraining the surface positions to the deformations given 
		// by the example poses.
		m_exPoses.clear();
		for (PDPositions& exPos : exPoses) {
			m_exPoses.push_back(extendSurfaceDeformationToTets(exPos));
		}
		PDScalar weight = 0.;//1. / (m_exPoses.size() + 1);
		std::vector<PDScalar> weights;
		for (unsigned int i = 0; i < exPoses.size(); i++) weights.push_back(weight);

		if (forSprings) {
			makeEdges();
			for (auto& edgex : m_edges) {
				int v1Ind = edgex.first;
				int v2Ind = edgex.second;
				PD3dVector edge = m_positions.row(v1Ind) - m_positions.row(v2Ind);
				ProjDynConstraint* c = new SpringConstraint(
					m_numVertices, v1Ind, v2Ind, edge.norm(), generalWeight * m_normalization,
					1, 1, m_exPoses, weights);
				m_springConstraints.push_back(c);
				addConstraint(c);
			}
		}
		else {
			for (unsigned int t = 0; t < m_numTets; t++) {
				// Add example pose constraint
				TetExampleBased* tec = new TetExampleBased(m_numVertices, t, m_tetrahedrons, m_positions, m_exPoses, weights, generalWeight * m_normalization);
				addConstraint(tec);
				m_tetExConstraints.push_back(tec);
			}
		}
	}
}
*/

PDPositions PD::ProjDynSimulator::extendSurfaceDeformationToTets(PDPositions & surfacePos)
{
	int numInnerVerts = m_numVertices - m_numOuterVertices;
	if (!m_surfDefExtInit) {
		// Set up the constraints, the matrices and the solver for surface
		// deformation extensions
		std::vector<Eigen::Triplet<double>> s1Entries;
		std::vector<Eigen::Triplet<double>> s2Entries;
		s1Entries.reserve(m_numTets * 12);
		s2Entries.reserve(m_numTets * 12);
		for (unsigned int t = 0; t < m_numTets; t++) {
			TetStrainConstraint* tc = new TetStrainConstraint(m_numVertices, t, m_tetrahedrons, m_positions, 1, 1, m_normalization);
			m_surfDefExtConstraints.push_back(tc);
			PDSparseMatrixRM& sel = tc->getSelectionMatrix();
			PDScalar weight = std::sqrt(tc->getWeight());
			for (int k = 0; k < sel.outerSize(); ++k) {
				for (PDSparseMatrixRM::InnerIterator it(sel, k); it; ++it) {
					if (it.col() < m_numOuterVertices) {
						s1Entries.push_back(Eigen::Triplet<PDScalar>(t * 3 + it.row(), it.col(), it.value() * weight));
					}
					else {
						s2Entries.push_back(Eigen::Triplet<PDScalar>(t * 3 + it.row(), it.col() - m_numOuterVertices, it.value() * weight));
					}
				}
			}
		}

		PDSparseMatrix S1(m_numTets * 3, m_numOuterVertices);
		PDSparseMatrix S2(m_numTets * 3, numInnerVerts);
		S1.setFromTriplets(s1Entries.begin(), s1Entries.end());
		S2.setFromTriplets(s2Entries.begin(), s2Entries.end());
		m_surfDefExtRHSMat = S2.transpose();
		PDSparseMatrix lhs = m_surfDefExtRHSMat * S2;
		m_surfDefExtSolver.compute(lhs);
		m_surfDefExtFixedPartMat = S1;
	}

	// Compute the fixed part of the rhs (which is S2^T * (p - S1 * pos_outer))
	PDPositions rhsFixedPart = m_surfDefExtFixedPartMat * surfacePos;
	PDPositions innerVerts(numInnerVerts, 3);
	// Initial guess: set all to zero
	innerVerts.setZero();

	int numIterations = 100;
	PDPositions fullPos(m_numVertices, 3);
	PDPositions auxils(m_numTets * 3, 3);
	PDPositions rhs(numInnerVerts, 3);
	fullPos.block(0, 0, m_numOuterVertices, 3) = surfacePos;
	for (unsigned int it = 0; it < numIterations; it++) {
		// Local step:
		// First attach the inner and outer vertex positions together to evaluate projections
		fullPos.block(m_numOuterVertices, 0, numInnerVerts, 3) = innerVerts;
		// Then evaluate all projections in parallel
		int didCollide = -1;
		PROJ_DYN_PARALLEL_FOR
			for (int i = 0; i < m_surfDefExtConstraints.size(); i++) {
				auxils.block(i * 3, 0, 3, 3) = m_surfDefExtConstraints[i]->getP(fullPos, didCollide) * m_surfDefExtConstraints[i]->getWeight();
			}

		// Global step:
		// Form rhs:
		rhs = m_surfDefExtRHSMat * (auxils - rhsFixedPart);
		// Solve system
		PROJ_DYN_PARALLEL_FOR
			for (int d = 0; d < 3; d++) {
				innerVerts.col(d) = m_surfDefExtSolver.solve(rhs.col(d));
			}
	}

	fullPos.block(m_numOuterVertices, 0, numInnerVerts, 3) = innerVerts;

	PD::storeMesh(fullPos, m_triangles, "D:\\surfDefExt.obj");

	return fullPos;
}
/*
void PD::ProjDynSimulator::setExampleWeights(std::vector<PDScalar>& exWeights)
{
	if (m_rhsInterpolation) {
		for (auto& g : m_snapshotGroups) {
			if (g.getName() == "tetex" || g.getName() == "spring") {
				g.setExampleWeights(exWeights);
			}
		}
	}
	else {
		for (ProjDynConstraint* tec : m_tetExConstraints) {
			((TetExampleBased*)tec)->setExampleWeights(exWeights);
		}
	}
}
*/

void PD::ProjDynSimulator::updateUsedVertices()
{
	std::cout << "Creating a list of used vertices and adapting constraints to this list..." << std::endl;
	
	m_usedVertexMap = new int[m_numVertices];
	
	for (unsigned int i = 0; i < m_numVertices; i++) m_usedVertexMap[i] = -1;
	
	m_usedVertices.clear();
	for (unsigned int v : m_additionalUsedVertices) {
		m_usedVertices.push_back(v);
		
	}
	for (ProjDynConstraint* c : m_sampledConstraints) {
		bool first = true;
		PDSparseMatrix sel = c->getSelectionMatrixTransposed();
		
		for (int k = 0; k < sel.outerSize(); ++k) {
			for (PDSparseMatrix::InnerIterator it(sel, k); it; ++it)
			{
				if (it.value() != PDScalar(0)) {
					m_usedVertices.push_back(it.row());
					
					if (first) {
						m_usedVerticesSlim.push_back(it.row());
						first = false;
						
					}
				}
			}
		}
	}
	
	std::sort(m_usedVertices.begin(), m_usedVertices.end());
	m_usedVertices.erase(std::unique(m_usedVertices.begin(), m_usedVertices.end()), m_usedVertices.end());

	std::sort(m_usedVerticesSlim.begin(), m_usedVerticesSlim.end());
	m_usedVerticesSlim.erase(std::unique(m_usedVerticesSlim.begin(), m_usedVerticesSlim.end()), m_usedVerticesSlim.end());
	
	
	
	m_slimToUsedIndices.resize(m_usedVerticesSlim.size());
	for (unsigned int i = 0; i < m_usedVerticesSlim.size(); i++) {
		auto const& it = std::find(m_usedVertices.begin(), m_usedVertices.end(), m_usedVerticesSlim[i]);
		if (it == m_usedVertices.end()) {
			std::cout << "Error while slimming used vertices..." << std::endl;
		}
		else {
			m_slimToUsedIndices[i] = *it;   
		}
	}

	for (unsigned int i = 0; i < m_usedVertices.size(); i++) {
		m_usedVertexMap[m_usedVertices[i]] = i;
	}

	m_positionsUsedVs.setZero(m_usedVertices.size(), 3);
	m_positionCorrectionsUsedVs.setZero(m_usedVertices.size(), 3);
	m_velocitiesUsedVs.setZero(m_usedVertices.size(), 3);
	
	if(m_usingSkinSubspaces){
		PDMatrix A(m_usedVertices.size(), m_baseFunctions.cols());
		unsigned int i = 0;
		for (unsigned int v : m_usedVertices) {
			A.row(i) = m_baseFunctions.row(v);
			i++;
		}
		m_usedVertexInterpolatorRHSMatrix = A.transpose();
		PDMatrix lhsMat = A.transpose() * A;
		
		Eigen::LLT<Eigen::MatrixXd> lltOfA(lhsMat); // compute the Cholesky decomposition of A
		if(lltOfA.info() == Eigen::NumericalIssue)
		{
		std::cout<< "Possibly dealing with non semi-positive definitie matrix!.. The current solvers might not work!" << std::endl;
		} 
		
		m_usedVertexInterpolator.compute(lhsMat);

		if (m_usedVertexInterpolator.info() != Eigen::Success) {
			std::cout << "Warning: Factorization of lhs matrix for used vertex interoplation was not successful!" << std::endl;
		}

		if (m_useSparseMatricesForSubspace) {
			m_usedVertexInterpolatorRHSMatrixSparse = m_usedVertexInterpolatorRHSMatrix.sparseView(0, PROJ_DYN_SPARSITY_CUTOFF);
			PDSparseMatrix lhsMatSparse = lhsMat.sparseView(0, PROJ_DYN_SPARSITY_CUTOFF_HIGH_PREC);
			m_usedVertexInterpolatorSparse.compute(lhsMatSparse);
			if (m_usedVertexInterpolatorSparse.info() != Eigen::Success) {
				std::cout << "Warning: Factorization of the SPARSE lhs matrix for used vertex interoplation was not successful!" << std::endl;
			}
		}

		for (ProjDynConstraint* c : m_sampledConstraints) {
			c->setUsedVertices(m_usedVertices);
		}

	#ifdef PROJ_DYN_USE_CUBLAS
		if (m_usedVertexUpdater != nullptr) {
			delete m_usedVertexUpdater;
		}
		if (m_rhsEvaluator != nullptr) {
			delete m_rhsEvaluator;
		}
		m_projUsedVerts.setZero(m_usedVertices.size(), m_baseFunctions.cols());
		for (int i = 0; i < m_usedVertices.size(); i++) {
			m_projUsedVerts.row(i) = m_baseFunctions.row(m_usedVertices[i]);
		}
		m_usedVertexUpdater = new CUDAMatrixVectorMultiplier(m_projUsedVerts);
		m_curTempVec.setZero(m_usedVertices.size());

		m_rhsEvalMat = m_projUsedVerts.transpose();
		m_rhsEvaluator = new CUDAMatrixVectorMultiplier(m_rhsEvalMat);

		if (m_useSparseMatricesForSubspace) {
			if (m_usedVertexUpdaterSparse != nullptr) {
				delete m_rhsEvaluator;
			}
			PDSparseMatrix sparseRhsEvalMat = m_projUsedVerts.sparseView(0, 1e-12);
			m_usedVertexUpdaterSparse = new CUSparseMatrixVectorMultiplier(sparseRhsEvalMat);
		}
	#else
		// Collect non-zero entries required to evaluate 
		m_usedVerticesBase.resize(m_usedVertices.size());
		m_usedVerticesBaseNNZ.resize(m_usedVertices.size());
		for (int v = 0; v < m_usedVertices.size(); v++) {
			int nnz = 0;
			std::vector<PDScalar> entries;
			std::vector<unsigned int> inds;
			for (int u = 0; u < m_baseFunctions.cols(); u++) {
				if (abs(m_baseFunctions(m_usedVertices[v], u)) > PROJ_DYN_SPARSITY_CUTOFF) {
					entries.push_back(m_baseFunctions(m_usedVertices[v], u));
					inds.push_back(u);
					nnz++;
				}
			}
			m_usedVerticesBase[v] = entries;
			m_usedVerticesBaseNNZ[v] = inds;
		}
	#endif
	}
	else if(m_usePosSnapBases){
	
		PDMatrix Ax(m_usedVertices.size(), m_baseXFunctions.cols());
		PDMatrix Ay(m_usedVertices.size(), m_baseYFunctions.cols());
		PDMatrix Az(m_usedVertices.size(), m_baseZFunctions.cols());
		
		
		unsigned int i = 0;
		for (unsigned int v : m_usedVertices) {
			Ax.row(i) = m_baseXFunctions.row(v);
			i++;
		}
		i = 0;
		for (unsigned int v : m_usedVertices) {
			Ay.row(i) = m_baseYFunctions.row(v);
			i++;
		}
		i = 0;
		for (unsigned int v : m_usedVertices) {
			Az.row(i) = m_baseZFunctions.row(v);
			i++;
		}
		
		m_usedVertexXInterpolatorRHSMatrix = Ax.transpose();
		PDMatrix lhsMatX = Ax.transpose() * Ax;
		
		m_usedVertexYInterpolatorRHSMatrix = Ay.transpose();
		PDMatrix lhsMatY = Ay.transpose() * Ay;
		
		m_usedVertexZInterpolatorRHSMatrix = Az.transpose();
		PDMatrix lhsMatZ = Az.transpose() * Az;
		 

		//----------------------------------------------------------------------------------------------------------------------------
		#pragma omp parallel
		#pragma omp single nowait
		{
		#pragma omp task
			m_usedVertexXInterpolator.compute(lhsMatX);

			if (m_usedVertexXInterpolator.info() != Eigen::Success) {
				std::cout << "Warning: Factorization of lhs X matrix for used vertex interoplation was not successful!" << std::endl;
			}

			if (m_useSparseMatricesForSubspace) {
				m_usedVertexXInterpolatorRHSMatrixSparse = m_usedVertexXInterpolatorRHSMatrix.sparseView(0, PROJ_DYN_SPARSITY_CUTOFF);
				PDSparseMatrix lhsMatXSparse = lhsMatX.sparseView(0, PROJ_DYN_SPARSITY_CUTOFF_HIGH_PREC);
				m_usedVertexXInterpolatorSparse.compute(lhsMatXSparse);
				if (m_usedVertexXInterpolatorSparse.info() != Eigen::Success) {
					std::cout << "Warning: Factorization of the SPARSE lhs X matrix for used vertex interoplation was not successful!" << std::endl;
				}
			}
		#pragma omp task
			m_usedVertexYInterpolator.compute(lhsMatY);

			if (m_usedVertexYInterpolator.info() != Eigen::Success) {
				std::cout << "Warning: Factorization of lhs Y matrix for used vertex interoplation was not successful!" << std::endl;
			}

			if (m_useSparseMatricesForSubspace) {
				m_usedVertexYInterpolatorRHSMatrixSparse = m_usedVertexYInterpolatorRHSMatrix.sparseView(0, PROJ_DYN_SPARSITY_CUTOFF);
				PDSparseMatrix lhsMatYSparse = lhsMatY.sparseView(0, PROJ_DYN_SPARSITY_CUTOFF_HIGH_PREC);
				m_usedVertexYInterpolatorSparse.compute(lhsMatYSparse);
				if (m_usedVertexYInterpolatorSparse.info() != Eigen::Success) {
					std::cout << "Warning: Factorization of the SPARSE lhs Y matrix for used vertex interoplation was not successful!" << std::endl;
				}
			}
		#pragma omp task
			m_usedVertexZInterpolator.compute(lhsMatZ);

			if (m_usedVertexZInterpolator.info() != Eigen::Success) {
				std::cout << "Warning: Factorization of lhs Z matrix for used vertex interoplation was not successful!" << std::endl;
			}

			if (m_useSparseMatricesForSubspace) {
				m_usedVertexZInterpolatorRHSMatrixSparse = m_usedVertexZInterpolatorRHSMatrix.sparseView(0, PROJ_DYN_SPARSITY_CUTOFF);
				PDSparseMatrix lhsMatZSparse = lhsMatZ.sparseView(0, PROJ_DYN_SPARSITY_CUTOFF_HIGH_PREC);
				m_usedVertexZInterpolatorSparse.compute(lhsMatZSparse);
				if (m_usedVertexZInterpolatorSparse.info() != Eigen::Success) {
					std::cout << "Warning: Factorization of the SPARSE lhs Z matrix for used vertex interoplation was not successful!" << std::endl;
				}
			}
		}
		//------------------------------------------------------------------------------------------------------------------------------

		for (ProjDynConstraint* c : m_sampledConstraints) {
			c->setUsedVertices(m_usedVertices);
		}

	#ifdef PROJ_DYN_USE_CUBLAS
			
		std::cout << "NOT yet availablr for POD spaces" << std::endl;
		return;
	
	#else
		// Collect non-zero entries required to evaluate 
		m_usedVerticesXBase.resize(m_usedVertices.size());
		m_usedVerticesXBaseNNZ.resize(m_usedVertices.size());
		
		m_usedVerticesYBase.resize(m_usedVertices.size());
		m_usedVerticesYBaseNNZ.resize(m_usedVertices.size());
		
		m_usedVerticesZBase.resize(m_usedVertices.size());
		m_usedVerticesZBaseNNZ.resize(m_usedVertices.size());
		//-------------------------------------------------------------------------------------------------------
		PROJ_DYN_PARALLEL_FOR
			for (int v = 0; v < m_usedVertices.size(); v++) {
				int nnz = 0;
				std::vector<PDScalar> entriesX;
				std::vector<unsigned int> indsX;
				for (int u = 0; u < m_baseXFunctions.cols(); u++) {
					if (abs(m_baseXFunctions(m_usedVertices[v], u)) > PROJ_DYN_SPARSITY_CUTOFF) {
						entriesX.push_back(m_baseXFunctions(m_usedVertices[v], u));
						indsX.push_back(u);
						nnz++;
					}
				}
				m_usedVerticesXBase[v] = entriesX;
				m_usedVerticesXBaseNNZ[v] = indsX;
			}
			
		PROJ_DYN_PARALLEL_FOR
			for (int v = 0; v < m_usedVertices.size(); v++) {
				int nnz = 0;
				std::vector<PDScalar> entriesY;
				std::vector<unsigned int> indsY;
				for (int u = 0; u < m_baseYFunctions.cols(); u++) {
					if (abs(m_baseYFunctions(m_usedVertices[v], u)) > PROJ_DYN_SPARSITY_CUTOFF) {
						entriesY.push_back(m_baseYFunctions(m_usedVertices[v], u));
						indsY.push_back(u);
						nnz++;
					}
				}
				m_usedVerticesYBase[v] = entriesY;
				m_usedVerticesYBaseNNZ[v] = indsY;
			}
		
		PROJ_DYN_PARALLEL_FOR
			for (int v = 0; v < m_usedVertices.size(); v++) {
				int nnz = 0;
				std::vector<PDScalar> entriesZ;
				std::vector<unsigned int> indsZ;
				for (int u = 0; u < m_baseZFunctions.cols(); u++) {
					if (abs(m_baseZFunctions(m_usedVertices[v], u)) > PROJ_DYN_SPARSITY_CUTOFF) {
						entriesZ.push_back(m_baseZFunctions(m_usedVertices[v], u));
						indsZ.push_back(u);
						nnz++;
					}
				}
				m_usedVerticesZBase[v] = entriesZ;
				m_usedVerticesZBaseNNZ[v] = indsZ;
			}
		//-----------------------------------------------------------------------------------------------------------
	#endif
	}

	m_constraintSamplesChanged = false;
}


PDMatrix PD::ProjDynSimulator::createSkinningWeights(unsigned int numSamples, PDScalar rMultiplier) {
	std::cout << "Choosing samples and comuting weights" << std::endl;
	std::vector<unsigned int> samples = m_sampler.getSamples(numSamples);
	PDScalar furthestDist = m_sampler.getSampleDiameter(samples);
	PDScalar r = furthestDist * rMultiplier;
	PDMatrix weightMat = m_sampler.getRadialBaseFunctions(samples, true, r);
	
	return weightMat;
}

void PD::ProjDynSimulator::createPositionSubspace(unsigned int numSamples, bool useSkinningSpace, bool usePositionPODSpace) 
{

	if (useSkinningSpace && !usePositionPODSpace) {
		
		StopWatch samplingT(10, 10);

		samplingT.startStopWatch();
		m_samples = m_sampler.getSamples(numSamples);
		samplingT.stopStopWatch();
		std::cout << "	Time to choose samples: " << (samplingT.lastMeasurement() / 1000000.0) << "s" << std::endl;

		numSamples = m_samples.size();
		std::sort(m_samples.begin(), m_samples.end());
		m_samples.erase(std::unique(m_samples.begin(), m_samples.end()), m_samples.end());

		PDScalar furthestDist = m_sampler.getSampleDiameter(m_samples);
		PDScalar r = furthestDist * m_baseFunctionRadius;

		samplingT.startStopWatch();
		m_baseFunctionWeights = m_sampler.getRadialBaseFunctions(m_samples, true, r);
		samplingT.stopStopWatch();
		std::cout << "	Time to compute weights: " << (samplingT.lastMeasurement() / 1000000.0) << "s" << std::endl;

		bool isFlat = false;
		if (m_positions.col(2).norm() < 1e-10) isFlat = true;
		m_baseFunctions = createSkinningSpace(m_positions, m_baseFunctionWeights, nullptr, 1U, nullptr, nullptr, isFlat);
		//m_basefunctions (m_numVertices, 4*numSamples -3)
	}
	else if (usePositionPODSpace && !useSkinningSpace){
		// read basis form a binary file
		PDMatrix Upca;
		// read the .bin file according to the pr-edefined number of components in (main.cpp/ doubleHRPD)
		if(m_usingSPLOCSPosSubspaces && !m_usingPODPosSubspaces){
			if (PD::loadBaseBinary(m_SPLOCSBasesDir + "K" + std::to_string(numSamples) + ".bin", Upca)) {
				std::cout << "SPLOCS basis size is (" << Upca.rows() << "," << Upca.cols() << ")" << std::endl;
			}
		}
		else if (m_usingPODPosSubspaces && !m_usingSPLOCSPosSubspaces){
			if (PD::loadBaseBinary(m_PCABasesDir + "K" + std::to_string(numSamples) + ".bin", Upca)) {
				std::cout << "POD basis size is (" << Upca.rows() << "," << Upca.cols() << ")" << std::endl;
			}
		}
		
		m_snapshotsBasesTmp = Upca;
		//std::cout.precision(17);
		
		if(Upca.cols() != 3*numSamples){
			std::cout << "Error: dimension of of basisFunctions not matching number of POD modes!" << std::endl;
			return;
		}
	}
	else if (usePositionPODSpace && useSkinningSpace){
		std::cout << "usePositionPODSpace && useSkinningSpace!" << std::endl;
		
	}
	
	else { 
		std::cout << "Error: no basis method was defined!" << std::endl;
	}

}

void PD::ProjDynSimulator::loadQDEIMnonlinearSubspace(int numQDEIMModes, bool useSkinningSpace, bool usePositionPODSpace, bool rhsInterpolation){
	if(!useSkinningSpace && !rhsInterpolation){
		PDMatrix M, S, B;
		if (PD::loadBaseBinary("/home/shaimaa/libigl/tutorial/doubleHRPD/basisExperimentedwithPD/deim/pWeightedDeimBasis_BunnyF400K"+std::to_string(numQDEIMModes)+".bin", M)) {
			
			tempMqdeim = M;
			std::cout << "V (P^T V)^{-1} = M, qdeim matrix binary file has been found, matrix has been loaded";
			std::cout << "with size ("<< tempMqdeim.rows() << "," << tempMqdeim.cols() << ")" << std::endl;
		}
		else{
			std::cout << "M qdeim binary could not be loaded" << std::endl;
		}
		if (PD::loadBaseBinary("/home/shaimaa/libigl/tutorial/doubleHRPD/basisExperimentedwithPD/deim/pWightedDeimSelMatPoints_BunnyF400K"+std::to_string(numQDEIMModes)+".bin", S)) {
			tempSqdeim = S;
			std::cout << "S qdeim matrix binary file has been found, matrix has been loaded" ;
			std::cout << " with size ("<< tempSqdeim.rows()  << ","<< tempSqdeim.cols() << ")" << std::endl;
			//std::cout << tempSqdeim;
			
		}
		else{
			std::cout << "S qdeim binary could not be loaded" << std::endl;
		}
		if (PD::loadBaseBinary("/home/shaimaa/libigl/tutorial/doubleHRPD/basisExperimentedwithPD/deim/pWightedDeimInterpolBlocksBunnyF400K"+std::to_string(numQDEIMModes)+".bin", B)) {
			qdeimBlocks = B;
			std::cout << "qdeim interpolation blockes binary file has been found, blockes indecies have been loaded" ;
			std::cout << " with size ("<< qdeimBlocks.rows()  << ","<< qdeimBlocks.cols() << ")" << std::endl;
			//std::cout << tempSqdeim;
			
		}
		else{
			std::cout << "qdeim/deim Blockes indecies binary could not be loaded" << std::endl;
		}
		
		
	}
	else{
		std::cout << "QDEIM nonlinear reduction can be used only in combination with POD for pos subspace" << std::endl;
		return;
	}

	//std::cout << "QDEIM nonlinear constarints projection subspace has been loaded " << std::endl;
}

void PD::ProjDynSimulator::setExternalForces(PDPositions fExt)
{
	m_fExt = fExt;
	recomputeWeightedForces();
}

void PD::ProjDynSimulator::addGravity(PDScalar gravity)
{
	m_fGravity.resize(m_numVertices, 3);
	for (int v = 0; v < m_numVertices; v++) {
		m_fGravity(v, 1) = -gravity * m_vertexMasses(v);
	}
	recomputeWeightedForces();
}

void PD::ProjDynSimulator::addFloor(int floorCoordinate, PDScalar floorHeight, PDScalar floorCollisionWeight)
{
	m_floorCoordinate = floorCoordinate;
	m_floorHeight = floorHeight;
	m_floorCollisionWeight = floorCollisionWeight;
}

void PD::ProjDynSimulator::addRandomFloor(PDScalar a, PDScalar b, PDScalar c, PDScalar d, PDScalar floorCollisionWeight)
{
	// The florr is the 3D plane: ax + by + cz+ d =0
	coeffX = a;
	coeffY = b;
	coeffZ = c;
	planeScalar = d;
	m_floorCollisionWeight = floorCollisionWeight;
}

void PD::ProjDynSimulator::setFrictionCoefficient(PDScalar coeff, PDScalar rCoeff) {
	m_frictionCoeff = coeff;
	if (rCoeff < 0) {
		m_repulsionCoeff = coeff;
	}
	else {
		m_repulsionCoeff = rCoeff;
	}
}


void PD::ProjDynSimulator::addEdgeSprings(PDScalar weight, PDScalar rangeMin, PDScalar rangeMax)
{
	makeEdges();

	for (auto& edgex : m_edges) {

		int v1Ind = edgex.first;
		int v2Ind = edgex.second;

		PD3dVector edge = m_positions.row(v1Ind) - m_positions.row(v2Ind);
		ProjDynConstraint* c = new SpringConstraint(
			m_numVertices, v1Ind, v2Ind, edge.norm(), weight * m_normalization,
			rangeMin, rangeMax);
		m_springConstraints.push_back(c);
		addConstraint(c);

	}
}

void PD::ProjDynSimulator::addTriangleStrain(PDScalar weight, PDScalar rangeMin, PDScalar rangeMax)
{
	for (int t = 0; t < m_numTriangles; t++) {
		StrainConstraint* sc = new StrainConstraint(m_numVertices, t, m_triangles, m_positions, rangeMin, rangeMax, weight * m_normalization);
		addConstraint(sc);
		m_strainConstraints.push_back(sc);
	}
}

void PD::ProjDynSimulator::addTetStrain(PDScalar weight, PDScalar rangeMin, PDScalar rangeMax)
{
	//_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	if (m_hasTetrahedrons) {  // add tetcConstrainttProj at each tet
		for (int t = 0; t < m_numTets; t++) {
			PDScalar avgX = 1; // (m_positions(m_tetrahedrons(t, 0), 0) + m_positions(m_tetrahedrons(t, 1), 0) + m_positions(m_tetrahedrons(t, 2), 0) + m_positions(m_tetrahedrons(t, 3), 0)) / 4.;
			TetStrainConstraint* sc = new TetStrainConstraint(m_numVertices, t, m_tetrahedrons, m_positions, rangeMin, rangeMax, (weight * m_normalization) / (avgX < 0 ? 10. : 1.));
			addConstraint(sc);
			m_tetStrainConstraints.push_back(sc);
			delete sc;
			
		}
	}
	else {
		std::cout << "Could not add tet-strain, no tetrahedrons available!" << std::endl;
	}

}

void PD::ProjDynSimulator::addBendingConstraints(PDScalar weight, bool preventBendingFlips, bool flatBending)
{
	//std::cout << "===PD::ProjDynSimulator::addBendingConstraints===" << std::endl;      //can be visited in pre-computations
	std::vector< std::vector< Edge > >& vertexStars = m_vertexStars;

	PDVector vertexTriMasses = vertexMasses(m_triangles, m_positions);

	for (int v = 0; v < m_numVertices; v++) {
		bool dontAdd = false;
		if (vertexStars.at(v).empty()) {
			dontAdd = true;
		}
		for (Edge& e : vertexStars.at(v)) {
			if (e.t2 < 0) {
				dontAdd = true;
			}
		}

		if (!dontAdd) {
			BendingConstraint* bc = new BendingConstraint(m_numVertices, v, m_positions, m_triangles, vertexStars.at(v), vertexTriMasses(v), weight * m_normalization, preventBendingFlips, flatBending);
			addConstraint(bc);
			m_bendingConstraints.push_back(bc);
		}

	}

	if (flatBending) m_flatBending = true;
}


PD::ProjDynSimulator::~ProjDynSimulator()    /// a destructor for the function
{
	/*
	for (auto c : m_constraints) {
	delete c;
	}
	if (m_floorConstraints != nullptr) {
	delete[] m_floorConstraints;
	}*/
	
}

/*
void PD::ProjDynSimulator::setParallelVUpdateBlockSize(int blocks)
{
	std::cout << "===PD::ProjDynSimulator::setParallelVUpdateBlockSize===" << std::endl;
	m_parallelVUpdateBSize = blocks;
	updateParallelVUpdateBlocks();
}

void PD::ProjDynSimulator::setEnableVUpdate(bool enable)
{
	std::cout << "===PD::ProjDynSimulator::setEnableVUpdate===" << std::endl;
	m_parallelVUpdate = enable;
}
*/
void PD::ProjDynSimulator::makeEdges()
{ // used for addEdgeSpring

	std::vector<std::vector<int>> visitedEdge;
	visitedEdge.resize(m_numVertices);

	if (m_hasTetrahedrons) {
		for (int t = 0; t < m_numTets; t++) {
			for (int e = 0; e < 4; e++) {
				int v1Ind = m_tetrahedrons(t, e), v2Ind = m_tetrahedrons(t, (e + 1) % 4);
				if (v1Ind > v2Ind) {
					int tmp = v1Ind;
					v1Ind = v2Ind;
					v2Ind = tmp;
				}

				if (std::find(visitedEdge[v1Ind].begin(), visitedEdge[v1Ind].end(), v2Ind) != visitedEdge[v1Ind].end()) {
					continue;
				}

				m_edges.push_back(std::pair<unsigned int, unsigned int>(v1Ind, v2Ind));
				visitedEdge[v1Ind].push_back(v2Ind);
			}
		}
	}
	else {
		for (int t = 0; t < m_numTriangles; t++) {
			for (int e = 0; e < 3; e++) {
				int v1Ind = m_triangles(t, e), v2Ind = m_triangles(t, (e + 1) % 3);
				if (v1Ind > v2Ind) {
					int tmp = v1Ind;
					v1Ind = v2Ind;
					v2Ind = tmp;
				}

				if (std::find(visitedEdge[v1Ind].begin(), visitedEdge[v1Ind].end(), v2Ind) != visitedEdge[v1Ind].end()) {
					continue;
				}

				m_edges.push_back(std::pair<unsigned int, unsigned int>(v1Ind, v2Ind));
				visitedEdge[v1Ind].push_back(v2Ind);
			}
		}
	}

}

void PD::ProjDynSimulator::updateParallelVUpdateBlocks()
{ // used in case skinSubspaces

	if (m_parallelVUpdate) {
		std::cout << "Setting up parallel blocks for vertex positions updates..." << std::endl;

		int blockSize = PROJ_DYN_VPOS_BLOCK_SIZE;
		int numBlocks = std::ceil((float)m_positions.rows() / (float)blockSize);
		m_baseFunctionsSparseBlocks.clear();
		for (int b = 0; b < numBlocks; b++) {
			int curSize = blockSize;
			if (b == numBlocks - 1) {
				curSize = blockSize - (numBlocks * blockSize - m_positions.rows());
			}
			PDSparseMatrixRM curBlock(curSize, m_baseFunctionsSparse.cols());
			curBlock = m_baseFunctionsSparse.block(b*blockSize, 0, curSize, m_baseFunctionsSparse.cols());
			m_baseFunctionsSparseBlocks.push_back(curBlock);
		}
	}
}
/*
void PD::ProjDynSimulator::initializeGPUVPosMapping(GLuint bufferId)
{
	std::cout << "===PD::ProjDynSimulator::initializeGPUVPosMapping===" << std::endl;
#ifdef PROJ_DYN_USE_CUBLAS
#ifdef ENABLE_DIRECT_BUFFER_MAP
	if (m_vPosGPUUpdate) {
		delete m_vPosGPUUpdate;
	}

	if (m_usingSkinSubspaces && m_baseFunctions.cols() > 0) {
		m_vPosGPUUpdate = new CUDAMatrixVectorMultiplier(m_baseFunctions);
		m_vPosGPUUpdate->setGLBuffer(bufferId);
	}
#endif
#endif
}

PDPositions & PD::ProjDynSimulator::getUsedVertexPositions()
{
	if (m_rhsInterpolation) {
		return m_positionsUsedVs;
	}
	else {
		return m_positions;
	}
}
*/

// used by each set of constraint in the simulations, case rhsInterpolation
void ProjDynSimulator::addConstraint(ProjDynConstraint* c, bool alwaysAdd) {
	m_isSetup = false;
	m_constraints.push_back(c);
	//if (m_rhsInterpolation){
		bool containsSample = false;
		if (!alwaysAdd) {
			if (c->getMainVertexIndex() > 0) {
				int vInd = c->getMainVertexIndex();
				if (std::find(m_constraintVertexSamples.begin(), m_constraintVertexSamples.end(), vInd) != m_constraintVertexSamples.end()) {
					containsSample = true;
				}
			}
			else if (c->getMainTriangleIndex() > 0) {
				int tInd = c->getMainTriangleIndex();
				if (std::find(m_constraintTriSamples.begin(), m_constraintTriSamples.end(), tInd) != m_constraintTriSamples.end()) {
					containsSample = true;
				}
			}
			else if (c->getMainTetIndex() > 0) {
				int tetInd = c->getMainTetIndex();
				if (std::find(m_constraintTetSamples.begin(), m_constraintTetSamples.end(), tetInd) != m_constraintTetSamples.end()) {
					containsSample = true;
				}
			}
		}
		if (containsSample || alwaysAdd) {
			addConstraintSample(c);   // add c to m_sampledConstraints
		}
		if (alwaysAdd) {
			m_additionalConstraints.push_back(c);
		}
	//}
}

void ProjDynSimulator::printTimeMeasurements() {
	
	std::cout << "===========================================================" << std::endl;

	if(m_usingSkinSubspaces){
	std::cout << "LBS reduction for position subspace" << std::endl;
	std::cout << m_numSamplesPosSubspace << " pos basis." << std::endl;
	
	}
	if(m_usePosSnapBases){
		if (m_usingSPLOCSPosSubspaces) {
			std::cout << "SPOLCS reduction for position subspace" << std::endl;
			std::cout << m_numPosSPLOCSModes << " components " << std::endl;
		}
		else {
			std::cout << "POD reduction for position subspace" << std::endl;
			std::cout << m_numPosPODModes << " components " << std::endl;
		}
	}
	if (m_rhsInterpolation) {
		std::cout << "LBS reduction constraints subspace" << std::endl;
		std::cout << m_rhsInterpolBaseSize << " constraints projection basis" << std::endl;

	}

	std::cout << "m_timeStep: " << m_timeStep << std::endl;
	std::vector< ProjDynConstraint* >* usedConstraints = &m_constraints;
	int numConstraints = usedConstraints->size();
	
	std::cout << "# of constraints: " << numConstraints << ", using " << m_usedVertices.size() << " of " << m_numVertices << " vertices." << std::endl;
	std::cout << "Time for precomputation: " << m_precomputationStopWatch.lastMeasurement() / 1000 << " milliseconds" << std::endl;
	PDScalar fps = 1000000. / (m_localStepStopWatch.evaluateAverage() * m_numIterations + m_globalStepStopWatch.evaluateAverage() * m_numIterations + m_updatingVPosStopWatch.evaluateAverage() * m_numIterations);
	PDScalar fpsD = 1000000. / (m_localStepStopWatch.evaluateAverage() * m_numIterations + m_globalStepStopWatch.evaluateAverage() * m_numIterations + m_updatingVPosStopWatch.evaluateAverage() * m_numIterations + m_fullUpdateStopWatch.evaluateAverage());
	std::cout << "FPS: " << fps << " (" << fpsD << ")" << std::endl;
	
	
	std::cout << "Average time for local step: " << m_localStepStopWatch.evaluateAverage() << " microseconds" << std::endl;
	std::cout << "Average time for global step: " << m_globalStepStopWatch.evaluateAverage() << " microseconds" << std::endl;
	
	std::cout << "Average time for only the constraint projection in the local step (includes summation if using rhs interpolation): " << m_localStepOnlyProjectStopWatch.evaluateAverage() << " microseconds" << std::endl;
	std::cout << "Average time for the rest of the local step: " << m_localStepRestStopWatch.evaluateAverage() << " microseconds" << std::endl;
	std::cout << "        ----> RHS-Reset/Summation/Projection+Moment. " << (m_localStepRestStopWatch.evaluateAverage() - m_momentumStopWatch.evaluateAverage() - m_constraintSummationStopWatch.evaluateAverage()) << "/" << m_constraintSummationStopWatch.evaluateAverage() << "/" << m_momentumStopWatch.evaluateAverage() << " microseconds" << std::endl;
#ifdef PROJ_DYN_USE_CUBLAS
	if (m_usingSkinSubspaces && m_rhsInterpolation) {
		std::cout << "        ----> Multiplication for Projection set/get/Dgemv: ";
		m_rhsEvaluator->printTimings();
		std::cout << std::endl;
	}
#endif

	std::cout << "Average time for global step: " << m_globalStepStopWatch.evaluateAverage() << " microseconds" << std::endl;
	std::cout << "Average time for updating relevant v. pos.: " << m_updatingVPosStopWatch.evaluateAverage() << " microseconds" << std::endl;
#ifdef PROJ_DYN_USE_CUBLAS
	if (m_usingSkinSubspaces) {
		std::cout << "        ----> Multiplication(set/get/Dgemv)/Ordering: " << (m_multiplicationForPosUpdate.evaluateAverage()) << "(";
		m_usedVertexUpdater->printTimings();
		std::cout << ")" << "/" << (m_sortingForPosUpdate.evaluateAverage()) << " (three multiplciations per inner it.)" << std::endl;
	}
#endif
	std::cout << "Average time for a full step with " << m_numIterations << " steps: " << m_totalStopWatch.evaluateAverage() << std::endl;
	std::cout << "Average time of only the surrounding block of a full step: " << m_surroundingBlockStopWatch.evaluateAverage() << std::endl;
	std::cout << "			----> Full vertex update: " << m_fullUpdateStopWatch.evaluateAverage() << std::endl;
	float totalAverage = m_totalStopWatch.evaluateAverage();
	float surroundingAverage = m_surroundingBlockStopWatch.evaluateAverage();
	float restTime = totalAverage - surroundingAverage;
	float localToGlobal = (float)m_localStepStopWatch.evaluateAverage() / (float)(m_globalStepStopWatch.evaluateAverage() + m_updatingVPosStopWatch.evaluateAverage() + m_localStepStopWatch.evaluateAverage());
	float localAverage = restTime * localToGlobal;
	float globalAverage = restTime * (1.f - localToGlobal);
	std::cout << "The local steps took " << (localAverage / totalAverage) << "% of the total time for a step." << std::endl;
	std::cout << "The global (incl. v. pos. upd.) steps took " << (globalAverage / totalAverage) << "% of the total time for a step." << std::endl;
	std::cout << "The surrounding block took " << (surroundingAverage / totalAverage) << "% of the total time for a step." << std::endl;
	std::cout << "		----> Full update of vertex positions: " << m_fullUpdateStopWatch.evaluateAverage() << std::endl;
	std::cout << "Total number of refactorizations: " << m_numRefactorizations << std::endl;
	std::cout << "===========================================================" << std::endl;
}


PDScalar PD::ProjDynSimulator::evaluateEnergy(PDPositions & q, PDPositions & s)
{
	PDPositions momVec = (q - s);
	/*
	for (unsigned int v = 0; v < m_numVertices; v++) {
	momVec.row(v) *= std::sqrt(m_vertexMasses(v));
	}
	PDScalar momE = momVec.norm();
	momE = momE * momE;
	*/ // here was a close for the commented part (shaimaa 20.01.22)
	PDScalar momE = (momVec.transpose() * m_massMatrix * momVec).trace();
	PDScalar innerE = 0;
	int dummyI = 0;
	PROJ_DYN_PARALLEL_FOR
		for (int i = 0; i < m_constraints.size(); i++) {
			ProjDynConstraint* c = m_constraints[i];
			PDPositions actualP = c->getSelectionMatrix() * q;
			PDPositions desiredP = c->getP(q, dummyI);
			PDScalar pNormSquared = (actualP - desiredP).norm();
			pNormSquared = pNormSquared * pNormSquared;
			innerE += (c->getWeight() / 2.) * pNormSquared;
		}
	PDScalar totE = (1. / (2. * m_timeStep * m_timeStep)) * momE + innerE;
	std::cout << "Mom. En. : " << momE << "; Elastic Potential: " << innerE << "; Total: " << totE << std::endl;
	return totE;
}

/*
void PD::ProjDynSimulator::setInitialPos(PDPositions & startPos)
{
	std::cout << "===PD::ProjDynSimulator::setInitialPos===" << std::endl;
	PDMatrix posMat;
	if (PD::loadBaseBinary(PD::getMeshFileName(m_meshURL, "_start.pos"), posMat)) {
		std::cout << "Found previously extended and possibly projected starting positions, loaded these..." << std::endl;
		m_positions = posMat;
		m_initialPos = m_positions;
	}
	else {
		std::cout << "Setting initial configuration..." << std::endl;

		if (!m_hasTetrahedrons) {
			m_positions = startPos;
		}
		else {
			std::cout << "	Extending initial surface configuration to interior..." << std::endl;
			m_positions = extendSurfaceDeformationToTets(startPos);
		}
		m_initialPos = m_positions;

	}
	if (m_usingSkinSubspaces) {
		std::cout << "	Projecting initial configuration to subspace..." << std::endl;
		projectToSubspace(m_positionsSubspace, m_positions, false);
		m_positions = m_baseFunctions * m_positionsSubspace;
		m_initialPos = m_positions;
		m_initialPosSub = m_positionsSubspace;
	}
	posMat = m_positions;
	PD::storeBaseBinary(posMat, PD::getMeshFileName(m_meshURL, "_start.pos"));
}
*/
void PD::ProjDynSimulator::setBlowup(PDScalar strength)
{
	m_blowupStrength = strength;
	for (auto& g : m_snapshotGroups) {
		if (g.getName() == "tetstrain" || g.getName() == "tetex") {
			g.setBlowup(strength);
		}
	}
}

void PD::ProjDynSimulator::addHandleConstraint(CenterConstraint * cc)
{
	m_handleConstraints.push_back(cc);
}

void PD::ProjDynSimulator::changeTimeStep(PDScalar newTimeStep)
{
	//std::cout << "===PD::ProjDynSimulator::changeTimeStep===" << std::endl;
	if (m_usingSkinSubspaces) {
		m_timeStep = newTimeStep;

		refreshLHS();

		m_rhsFirstTermMatrix = m_rhsFirstTermMatrixPre * (1. / (m_timeStep * m_timeStep));

		if (m_useSparseMatricesForSubspace) {
			m_rhsFirstTermMatrixSparse = m_rhsFirstTermMatrix.sparseView(0, PROJ_DYN_SPARSITY_CUTOFF_HIGH_PREC);
		}

		recomputeWeightedForces();
	}
	else {
		std::cout << "Changing time-steps is not supported for other than SkinSubspaces case!" << std::endl;
	}
}


void ProjDynSimulator::addAdditionalConstraints(PDPositions& pos, PDPositions& rhs, bool* collidedVertices) {

	if (!m_additionalConstraints.empty()) {

		int numAddConstraints = m_additionalConstraints.size();
		//std::cout << " m_additionalConstraints size -------------------- " << numAddConstraints << std::endl;
		if (m_additionalConstraintsAuxTemp.size() != numAddConstraints) {
			m_additionalConstraintsAuxTemp.resize(numAddConstraints);
		}

		if (m_usingSkinSubspaces) {
		
		PROJ_DYN_PARALLEL_FOR
			for (int j = 0; j < numAddConstraints; j++) {
				ProjDynConstraint* c = m_additionalConstraints.at(j);
				int didCollide = -1;
				PDPositions aux = c->getP(pos, didCollide);
				
				m_additionalConstraintsAuxTemp[j] = c->getSubspaceRHSMat(m_baseFunctionsTransposed) * (aux * c->getWeight()) * m_stiffnessFactor;   // U.T S.T p
				
				if (didCollide > 0) collidedVertices[didCollide] = true;
				
				PROJ_DYN_PARALLEL_FOR
					for (int d = 0; d < 3; d++) {
						for (int j = 0; j < numAddConstraints; j++) {
							rhs.col(d) += m_additionalConstraintsAuxTemp[j].col(d);
						}
					}
			}
		}
		else {
			PROJ_DYN_PARALLEL_FOR
				for (int j = 0; j < numAddConstraints; j++) {
					ProjDynConstraint* c = m_additionalConstraints.at(j);
					int didCollide = -1;
					PDPositions aux = c->getP(pos, didCollide);
		
					m_additionalConstraintsAuxTemp[j] = c->getSelectionMatrixTransposed() * (aux * c->getWeight())  * m_stiffnessFactor;  // S.T p
					
					if (didCollide > 0) collidedVertices[didCollide] = true;
					
					PROJ_DYN_PARALLEL_FOR
						for (int d = 0; d < 3; d++) {
							for (int j = 0; j < numAddConstraints; j++) {
								rhs.col(d) += m_additionalConstraintsAuxTemp[j].col(d);
						
							}
						}
				}
		}



	}
}

/* RHSInterpolationGroup is the fitting approximation of the RHS into the skinning subspace
   Initialized the rhs interpolation/fitting method. */
void PD::ProjDynSimulator::initRHSInterpolGroup(RHSInterpolationGroup& g, std::vector<unsigned int>& samples) {    // , PDMatrix& hessian) removed
	
	if (m_rhsInterpolReusableWeights.rows() <= 0) {
		std::cout << "Computing reusable weights for the rhs interpolation bases..." << std::endl;
		m_rhsInterpolReusableWeights = createSkinningWeights(m_rhsInterpolBaseSize * 0.5, m_rhsInterpolWeightRadiusMultiplier);
	}
	//std::cout << m_rhsInterpolReusableWeights.rows() << std::endl;
	if (m_rhsInterpolReusableWeights.hasNaN()) {
		std::cout << "Warning: NaN entries in reusable weights!" << std::endl;
	}
	PDMatrix weights;
	if (g.getName() == "bend" || g.getName() == "spring") {
		weights = m_rhsInterpolReusableWeights;  
	}
	else if (g.getName() == "strain") {
		std::cout << "Rescaling weights for triangles..." << std::endl;
		weights = PD::getTriangleWeightsFromVertexWeights(m_rhsInterpolReusableWeights, m_triangles);
	}
	else if (g.getName() == "tetstrain" || g.getName() == "tetex") {
		std::cout << "Rescaling weights for tets..." << std::endl;
		weights = PD::getTetWeightsFromVertexWeights(m_rhsInterpolReusableWeights, m_tetrahedrons);
	}
	if (weights.hasNaN()) {
		std::cout << "Warning: NaN entries in remapped reusable weights!" << std::endl;
	}
	g.createBasisViaSkinningWeights(m_rhsInterpolBaseSize, m_positions, weights, true, m_triangles, m_tetrahedrons);
	
	if (m_usingSkinSubspaces || m_rhsInterpolation) {

		g.initInterpolation(m_numVertices, samples, m_baseFunctionsTransposedSparse);    // compute an initialization for reduced constraint projection: \tilde{p}
	}
	else{	
		std::cout << "Error! initiating rhs interpolation is not yet available only with position reduction...se if you can fix: initInterpolationCaseNOPosReduction" << std::endl;
		return;
		
	}
}

void PD::ProjDynSimulator::initQDEIMRHSInterpolGroup(RHSInterpolationGroup& g, std::vector<unsigned int>& samples , std::vector<PDMatrix>& Mqdeim){
	//Initialized the rhs interpolation/fitting method for the QDEIM samples/basis .
	g.initQDEIMInterpolation(m_numVertices, samples, Mqdeim);
}
 
 
void ProjDynSimulator::setup() { 
	
	// we come here directly after printing mesh statistics
	std::cout << "Setting simulation..." << std::endl;
	//std::cout << m_positions << std::endl;  //  same initial m_positions for all

#ifndef EIGEN_DONT_PARALLELIZE
	Eigen::setNbThreads(PROJ_DYN_NUM_THREADS);
#endif 

	m_precomputationStopWatch.startStopWatch();

	m_positionCorrections.setZero(m_positions.rows(), 3);

	/* First, in case we use subspaces to reduce position, we create or load position subspace basis functions */
	if (m_usingPosSubspaces) {
		//bool loadSuccess = false;
		if (m_usingSkinSubspaces && !m_usePosSnapBases) {
			std::cout << "Creating subspaces..." << std::endl;
			createPositionSubspace(m_numSamplesPosSubspace,true, false); // pick wich true/false: do we useSkinningSpace? or usingPODPosSubSpace?
			finalizeBaseFunctions(); 
		}
		else if (m_usePosSnapBases && !m_usingSkinSubspaces){
			std::cout << "Loading subspaces..." << std::endl;
			if (m_usingPODPosSubspaces) {
				createPositionSubspace(m_numPosPODModes, false, true);  // here we choose usingPODPosSubSpace
				std::cout << "POD subspaces have been loaded..." << std::endl;
			}
			else if (m_usingSPLOCSPosSubspaces)
			{
				createPositionSubspace(m_numPosSPLOCSModes, false, true);  // here we choose usingPODPosSubSpace
				std::cout << "SPLOCS subspaces have been loaded..." << std::endl;
			}
			
			// In the POD case, different handling of basis are required, 
			// we decople the (X, Y, Z) dimensions and use three matrices so that we solve in parallel for each
			// slicing m_baseFunctions to m_baseXFunctions, m_baseYFunctions and m_baseYFunctions 
			// m_numPosPODModes+1: because we add the original mesh as a component too
			m_baseXFunctions.setZero(m_snapshotsBasesTmp.rows(), m_numPosPODModes+1);
			m_baseYFunctions.setZero(m_snapshotsBasesTmp.rows(), m_numPosPODModes+1);
			m_baseZFunctions.setZero(m_snapshotsBasesTmp.rows(), m_numPosPODModes+1);
			
			
			if(3*m_numPosPODModes != m_snapshotsBasesTmp.cols()){  
				std::cout << "Sizes are not matching... we have "<<  m_snapshotsBasesTmp.cols() <<" columns and 3*m_numPosPODModes = "<< 3*m_numPosPODModes <<  std::endl;
			}
			
			
			//PROJ_DYN_PARALLEL_FOR
			for (int k = 0; k < m_numPosPODModes; k++){
				//std::cout << normFactor << std::endl;
				for(int v = 0 ; v < m_numVertices; v++){
					// using different signs for the basis rotates the fist frame!
					m_baseXFunctions(v, k) = m_snapshotsBasesTmp(v, k);
					m_baseYFunctions(v, k) = m_snapshotsBasesTmp(v, m_numPosPODModes + k);
					m_baseZFunctions(v, k) = m_snapshotsBasesTmp(v, 2*m_numPosPODModes + k);
				}
			}
			// add the original mesh as the component
			for(int v = 0 ; v < m_numVertices; v++){
					m_baseXFunctions(v, m_numPosPODModes) = m_positions(v,0);
					m_baseYFunctions(v, m_numPosPODModes) = m_positions(v,1);
					m_baseZFunctions(v, m_numPosPODModes) = m_positions(v,2);
			}
							
				
			if (m_baseXFunctions.hasNaN() || m_baseYFunctions.hasNaN() || m_baseZFunctions.hasNaN()) {
				std::cout << "Error: NaN entries in POD basis matrixies" << std::endl;
			}
				
			finalizePODBaseFunctions();   // this function does finalize the three matrices
		}
		
	// Sparse subspace basis needs to be available before the snapshot groups get initialized
		if(m_useSparseMatricesForSubspace){
			std::cout << "Sparsifing POD base matrcies... " << std::endl;
			if(m_usingSkinSubspaces && !m_usePosSnapBases){
				// case Skinning subspaces
				m_baseFunctionsSparse = m_baseFunctions.sparseView(0, PROJ_DYN_SPARSITY_CUTOFF);
				m_baseFunctionsTransposedSparse = m_baseFunctionsTransposed.sparseView(0, PROJ_DYN_SPARSITY_CUTOFF);
			}
			if(m_usePosSnapBases && !m_usingSkinSubspaces){
			// case POD subspaces
				
				m_baseXFunctionsSparse = m_baseXFunctions.sparseView(0, PROJ_DYN_SPARSITY_CUTOFF);
				m_baseXFunctionsTransposedSparse = m_baseXFunctionsTransposed.sparseView(0, PROJ_DYN_SPARSITY_CUTOFF);
				
				m_baseYFunctionsSparse = m_baseYFunctions.sparseView(0, PROJ_DYN_SPARSITY_CUTOFF);
				m_baseYFunctionsTransposedSparse = m_baseYFunctionsTransposed.sparseView(0, PROJ_DYN_SPARSITY_CUTOFF);
				
				m_baseZFunctionsSparse = m_baseZFunctions.sparseView(0, PROJ_DYN_SPARSITY_CUTOFF);
				m_baseZFunctionsTransposedSparse = m_baseZFunctionsTransposed.sparseView(0, PROJ_DYN_SPARSITY_CUTOFF);				
			}
		}
		/*
		if(m_usingQDEIMComponents){

			// load and prepare the M and S mats to do nonlinear reduction
			loadQDEIMnonlinearSubspace(m_numQDEIMModes, m_usingSkinSubspaces, m_usingPODPosSubspaces, m_rhsInterpolation);
			int constSize = 0;
			int p = 1;
			if (m_TetStrainOnly) {
				p = 3;
				constSize = 3;
				}
			
			
			if(m_solveDeimLS){
				deimBasisCols = m_numQDEIMModes;
			}
			else{
				deimBasisCols = m_numQDEIMModes * p;
			}	
				
			//m_Mqdeim = V (P^T V)^{-1} 
			m_xMqdeim.setZero(tempMqdeim.rows(), deimBasisCols);
			m_yMqdeim.setZero(tempMqdeim.rows(), deimBasisCols);
			m_zMqdeim.setZero(tempMqdeim.rows(), deimBasisCols);
			
			
			m_SqdeimX.resize(m_numQDEIMModes);
			m_SqdeimY.resize(m_numQDEIMModes);
			m_SqdeimZ.resize(m_numQDEIMModes);
			
			if (3*deimBasisCols == tempMqdeim.cols() && m_numQDEIMModes == tempSqdeim.rows()){
			
			//PROJ_DYN_PARALLEL_FOR
				for (int k = 0; k < deimBasisCols; k++){
					//std::cout << k << std::endl;
					for(int v = 0 ; v < tempMqdeim.rows() ; v++){
						// filling only the required number of modes from the loded binary
						m_xMqdeim(v, k) = tempMqdeim(v, k);
						m_yMqdeim(v, k) = tempMqdeim(v, deimBasisCols + k);
						m_zMqdeim(v, k) = tempMqdeim(v, 2*deimBasisCols + k);
					}
				}
				
				
				m_Mqdeim.resize(3);
				m_Mqdeim[0] = m_xMqdeim;
				m_Mqdeim[1] = m_yMqdeim;
				m_Mqdeim[2] = m_zMqdeim;
				
				if(tempSqdeim.cols()==3){
					//PROJ_DYN_PARALLEL_FOR 
						// just for the names of the matrices to be consistent but this step is not necessary
						for (int k = 0; k < m_numQDEIMModes; k++){
							m_SqdeimX[k] = tempSqdeim(k, 0);
							m_SqdeimY[k] = tempSqdeim(k, 1);
							m_SqdeimZ[k] = tempSqdeim(k, 2);
						}
					m_Sqdeim.resize(3);
					m_Sqdeim[0] = m_SqdeimX;
					m_Sqdeim[1] = m_SqdeimY;
					m_Sqdeim[2] = m_SqdeimZ;								
				}		
				else{
					std::cout << "Fatal error! in Qdeim selection mat dimension" << std::endl;
					return;
				}

					std::cout << "QDEIM nonlinear subspaces have been loaded..." << std::endl;
				
			}
			else{
				std::cout << "FATAL ERROR!  in QDEIM matrices dimension" << std::endl;
				return;
			}
		
			m_xMqdeimSparse = m_xMqdeim.sparseView(0, PROJ_DYN_SPARSITY_CUTOFF);
			m_yMqdeimSparse = m_yMqdeim.sparseView(0, PROJ_DYN_SPARSITY_CUTOFF);
			m_zMqdeimSparse = m_zMqdeim.sparseView(0, PROJ_DYN_SPARSITY_CUTOFF);
				
		} */
	} // End of basis loading 
	
	/*  Second, if using subspaces, after finalizing the basis function (basis and basis.T ready!), we set up projection of full positions into the subspace.
	Initial subspace positions/velocities will be computed from full positions/velocities. */
	
	if(m_usingPosSubspaces){
		if (m_usingSkinSubspaces && !m_usePosSnapBases) {
			std::cout << "Projecting positions, velocities and forces into the Skinning subspace... " << std::endl;
			// We need to project the current positions to the subspace, which will be
			// done by solving a  least squares problem since the subspace is not assumed
			// to be orthonormal.
			PDMatrix L = m_baseFunctionsTransposed * m_massMatrix * m_baseFunctions;
			m_subspaceSolver.compute(L);

			m_positionsSubspace.setZero(m_baseFunctions.cols(), 3);
			m_velocitiesSubspace.setZero(m_baseFunctions.cols(), 3);

			projectToSubspace(m_positionsSubspace, m_positions, false);
			projectToSubspace(m_velocitiesSubspace, m_velocities, false);

			m_positions = m_baseFunctions * m_positionsSubspace;
			m_velocities = m_baseFunctions * m_velocitiesSubspace;
		}
		else if(m_usePosSnapBases && !m_usingSkinSubspaces){
			std::cout << "Prepapring subSpcaces and POD Subspaces solvers.... " << std::endl;
			// TODO: We need to project the current positions to the subspace, which can be done through matrix-vector product 
			// because in this case matrices are assumed to be orthonormal.
			
			if(!isPODBasisOrthogonal){
				m_baseXFunctionsSquared = m_baseXFunctionsTransposed * m_massMatrix * m_baseXFunctions;
				m_baseYFunctionsSquared = m_baseYFunctionsTransposed * m_massMatrix * m_baseYFunctions;
				m_baseZFunctionsSquared = m_baseZFunctionsTransposed * m_massMatrix * m_baseZFunctions;

				#pragma omp parallel
				#pragma omp single nowait
				{
				#pragma omp task
					m_subspaceXSolver.compute(m_baseXFunctionsSquared);
				#pragma omp task
					m_subspaceYSolver.compute(m_baseYFunctionsSquared);
				#pragma omp task
					m_subspaceZSolver.compute(m_baseZFunctionsSquared);
				}
				
				if(m_subspaceXSolver.info()!= Eigen::Success || m_subspaceYSolver.info()!= Eigen::Success  || m_subspaceZSolver.info()!= Eigen::Success  ) {
					// solving failed
					std::cout << "FATAL ERROR! subspaceSolvers for nonOrthogonal basis failed" << std::endl;
					return;
				}
				if(m_useSparseMatricesForSubspace){
					PDSparseMatrix m_baseXFunctionsSquaredSparse, m_baseYFunctionsSquaredSparse, m_baseZFunctionsSquaredSparse;
					m_baseXFunctionsSquaredSparse = m_baseXFunctionsTransposedSparse * m_massMatrix * m_baseXFunctionsSparse;
					m_baseYFunctionsSquaredSparse = m_baseYFunctionsTransposedSparse * m_massMatrix * m_baseYFunctionsSparse;
					m_baseZFunctionsSquaredSparse = m_baseZFunctionsTransposedSparse * m_massMatrix * m_baseZFunctionsSparse;

					#pragma omp parallel
					#pragma omp single nowait
					{
					#pragma omp task
						m_subspaceXSparseSolver.compute(m_baseXFunctionsSquaredSparse);
					#pragma omp task
						m_subspaceYSparseSolver.compute(m_baseYFunctionsSquaredSparse);
					#pragma omp task
						m_subspaceZSparseSolver.compute(m_baseZFunctionsSquaredSparse);
					}				
				}									
			}
			
			m_positionsSubspace.setZero(m_baseXFunctions.cols(), 3);
			m_velocitiesSubspace.setZero(m_baseXFunctions.cols(), 3);

			std::cout << "Projecting positions, velocities and forces into the POD subspace... " << std::endl;
						
			if(m_useSparseMatricesForSubspace){
				projectToSparsePODSubspace(m_positionsSubspace, m_positions, isPODBasisOrthogonal);   
				projectToSparsePODSubspace(m_velocitiesSubspace, m_velocities, isPODBasisOrthogonal);
			}
			else{
				// inialize subPos and subVeloceties from the fullPos and fullVeloceties
				projectToPODSubspace(m_positionsSubspace, m_positions, isPODBasisOrthogonal); 
				projectToPODSubspace(m_velocitiesSubspace, m_velocities, isPODBasisOrthogonal);
			}
			
			// Full = Basis * reduced: I do not think we need to do this with POD basis again, full positions is already the initial positions, or?
			//std::cout << "Updating full positions, velocities ... " << std::endl;
			
			//std::cout << "Update ready... " << std::endl;
		}
		
	}


	// Collect constraints for building global system and interpolation subspaces for rhs interpolation
	// (if no rhs interpolation is used, we simply collect all constraints, otherwise we only
	// use constraints from the main group and treat the rest as additional constraints) (?)
	
	/* the "m_constraints" are the ones added in main.cpp after the simulator has been initiated
	   example: sim->addTetStrain(0.00051, 1.f, 1.f); (in main.cpp) */
	std::vector< ProjDynConstraint* >* usedConstraints = &m_constraints;
	std::vector< ProjDynConstraint* > collectedConstraints;
		
	
	if (m_rhsInterpolation || (m_usingQDEIMComponents && m_solveDeimLS) ) {  // TODO: This case (m_usingQDEIMComponents && m_solveDeimLS) not yet tested.
		std::cout << "Collecting constraints for interpolation..." << std::endl;
		collectedConstraints.clear();
		for (ProjDynConstraint* c : m_bendingConstraints) {
			collectedConstraints.push_back(c);
		}
		
		for (ProjDynConstraint* c : m_strainConstraints) {
			collectedConstraints.push_back(c);
		}
		
		for (ProjDynConstraint* c : m_collisionConstraints) {
			collectedConstraints.push_back(c);
		}
		
		for (ProjDynConstraint* c : m_tetStrainConstraints) {
			collectedConstraints.push_back(c);
		}
		
		for (ProjDynConstraint* c : m_additionalConstraints) {
			collectedConstraints.push_back(c);
		}
		
		usedConstraints = &collectedConstraints;
		
		// Here we create a preliminary sampling of elements which are used
		// to choose which constraints should be evaluated.
		// These will be overwritten if constraint groups are used
		// that suggest DEIM samples.
		std::cout << "Sampling constraints..." ;
		if (m_rhsInterpolation ){
			createConstraintSampling(m_numConstraintSamples);	
		}
		else{
			
			createQDEIMConstraintTetStrainSampling();
			
		}
		   
		//std::cout << " DONE! " << std::endl;
	}

	

	// If using r.h.s. interpolation, build interpolation subspaces
	// and constraint sampling for each group.
	// The LHS matrix will also be built from these constraint interpolation groups.
	
	// Set up interpolation groups and adapt the lhs side matrix for using them
	if (m_rhsInterpolation || (m_usingQDEIMComponents && m_solveDeimLS) ) {
		std::cout << "Initiating snapshot groups for constraints ... " << std::endl;
		m_snapshotGroups.clear();
		// RHSInterpolationGroup initializes the full set of constraints 
		if (!m_springConstraints.empty()) {
			m_snapshotGroups.push_back(RHSInterpolationGroup("spring", m_springConstraints, m_positions,
				m_vertexMasses, m_triangles, m_tetrahedrons, m_rhsRegularizationWeight));
		}
		if (!m_bendingConstraints.empty() && !m_flatBending) {
			m_snapshotGroups.push_back(RHSInterpolationGroup("bend", m_bendingConstraints, m_positions,
				m_vertexMasses, m_triangles, m_tetrahedrons, m_rhsRegularizationWeight));
		}
		if (!m_strainConstraints.empty()) {
			m_snapshotGroups.push_back(RHSInterpolationGroup("strain", m_strainConstraints, m_positions,
				m_vertexMasses, m_triangles, m_tetrahedrons, m_rhsRegularizationWeight));
		}
		if (!m_tetStrainConstraints.empty()) {
			m_snapshotGroups.push_back(RHSInterpolationGroup("tetstrain", m_tetStrainConstraints, m_positions,
				m_vertexMasses, m_triangles, m_tetrahedrons, m_rhsRegularizationWeight));
		}
		if (!m_tetExConstraints.empty()) {
			m_snapshotGroups.push_back(RHSInterpolationGroup("tetex", m_tetExConstraints, m_positions,
				m_vertexMasses, m_triangles, m_tetrahedrons, m_rhsRegularizationWeight));
		}

		// If snapshot groups are used they need to be initialized
		if(m_rhsInterpolation){
			for (auto& g : m_snapshotGroups) {
				// Initialization of the group depends on the constraint that's being used
				// initialize the reduced constraints projection for each group
				if (g.getName() == "bend") {
					initRHSInterpolGroup(g, m_constraintVertexSamples); //, hessian);
				}
				else if (g.getName() == "spring") {
					initRHSInterpolGroup(g, m_constraintVertexSamples); //, hessian);
				}
				else if (g.getName() == "strain") {
					initRHSInterpolGroup(g, m_constraintTriSamples); //, hessian);
				}
				else if (g.getName() == "tetstrain" || g.getName() == "tetex") {
					initRHSInterpolGroup(g, m_constraintTetSamples); //, hessian);
				}
				else {
					std::cout << "ERROR: unknown rhs interpolation group: " << g.getName() << "!" << std::endl;
				}

				// Maintain list of constraints that have been sampled
				std::vector<ProjDynConstraint*>& sampledCons = g.getSampledConstraints();
				for (ProjDynConstraint* c : sampledCons) m_sampledConstraints.push_back(c);
				
				
			}
			
			std::cout << " LBS Snapshot groups for constraints has been initialized!" << std::endl;
		
		}
		else {  // if(m_usingQDEIMComponents && m_solveDeimLS)
			for (auto& g : m_snapshotGroups) {
				// Initialization of the group depends on the constraint that's being used
				// initialize the reduced constraints projection for each group
				
				if (g.getName() == "tetstrain" || g.getName() == "tetex") {
				
					//for (int m: m_constraintTetSamples) std::cout << m <<std::endl;
					initQDEIMRHSInterpolGroup(g, m_constraintTetSamples, m_Mqdeim); 
					
					
				}
				else{
					std::cout << "QDEIM Snapshot groups are not ready for this group!" << g.getName() << std::endl;
				}
				/*
				// TODO: currently QDEIM is implemented only for tet strains, we need to do the rest of the constarints groups
				else if (g.getName() == "spring") {
					initQDEIMRHSInterpolGroup(g, m_constraintVertexSamples); //, hessian);
				}
				else if (g.getName() == "strain") {
					initQDEIMRHSInterpolGroup(g, m_constraintTriSamples); //, hessian);
				}
				else if (g.getName() == "bend") {
					initQDEIMRHSInterpolGroup(g, m_constraintVertexSamples); //, hessian);
				}
				else {
					std::cout << "ERROR: unknown rhs interpolation group: " << g.getName() << "!" << std::endl;
				} */

				// Maintain list of constraints that have been sampled
				std::vector<ProjDynConstraint*>& sampledCons = g.getSampledConstraints();
							
				for (ProjDynConstraint* c : sampledCons) m_sampledConstraints.push_back(c);
			}
			
			std::cout << " QDEIM Snapshot groups for constraints has been initialized!" << std::endl;
		}
	}

	/* Initialize the LHS and RHS matrices for the global system: */
	
	// 1) Compute the momentum part of the lhs and rhs matrices of the global step
	// in case no position space reduction, or we run full simulation, the solver uses these terms (no projection required)
	std::cout << "Initiating momentum term of LHS and RHS matrices ..." << std::endl;
	m_lhsMatrix = m_massMatrix;
	m_lhsMatrix *= 1.f / (m_timeStep * m_timeStep);
	m_rhsMasses.setZero(m_numVertices);
	for (int v = 0; v < m_numVertices; v++) {
		m_rhsMasses(v) = m_vertexMasses(v) / (m_timeStep * m_timeStep);
	}

	/* And, if we use position spaces reduction, we project the momentum terms in bothe RHS and RHL to low dim subspaces*/
	
	if (m_usingSkinSubspaces&& !m_usePosSnapBases) {
		std::cout << "Projecting the momentum term RHS to skinning subspaces, for the global system ..." << std::endl;
		m_rhsFirstTermMatrixPre = m_baseFunctionsTransposed * m_massMatrix * m_baseFunctions;
		m_rhsFirstTermMatrix = m_rhsFirstTermMatrixPre * (1. / (m_timeStep * m_timeStep));    //m_rhsFirstTermMatrix = (U.T M U / h^2)
		rhs2.setZero(m_baseFunctions.cols(), 3);
		
		std::cout << "Projecting the momentum term LHS to skinning subspaces, for the global system ..." << std::endl;
		// Momentum term
		m_subspaceLHS_mom = m_baseFunctionsTransposed * m_lhsMatrix * m_baseFunctions;       // * (m_timeStep * m_timeStep);
		PDMatrix eps(m_subspaceLHS_mom.rows(), m_subspaceLHS_mom.rows());
		eps.setIdentity();
		eps *= 1e-10;
		m_subspaceLHS_mom += eps;   // m_subspaceLHS_mom = (U.T M U / h^2)
		
	}
	else if (m_usePosSnapBases && !m_usingSkinSubspaces) {
	
		rhsX2.setZero(m_baseXFunctions.cols(), 3);
		rhsY2.setZero(m_baseYFunctions.cols(), 3);
		rhsZ2.setZero(m_baseZFunctions.cols(), 3);
		
		std::cout << "Projecting the momentum term RHS matrices to POD subspaces, for the global system ..." << std::endl;
		
		if(isPODBasisOrthogonal){  //in the orthogonal case: U.T M U = Identity
			m_rhsXFirstTermMatrixPre.setIdentity(m_baseXFunctions.cols(), m_baseXFunctions.cols());
			m_rhsYFirstTermMatrixPre.setIdentity(m_baseYFunctions.cols(), m_baseYFunctions.cols());
			m_rhsZFirstTermMatrixPre.setIdentity(m_baseZFunctions.cols(), m_baseZFunctions.cols());	
			
			m_rhsXFirstTermMatrix.setZero(m_baseXFunctions.cols(), m_baseXFunctions.cols());
			m_rhsYFirstTermMatrix.setZero(m_baseYFunctions.cols(), m_baseYFunctions.cols());
			m_rhsZFirstTermMatrix.setZero(m_baseZFunctions.cols(), m_baseZFunctions.cols());
			
		}
		else{
			m_rhsXFirstTermMatrixPre.setZero(m_baseXFunctions.cols(), m_baseXFunctions.cols());
			m_rhsYFirstTermMatrixPre.setZero(m_baseYFunctions.cols(), m_baseYFunctions.cols());
			m_rhsZFirstTermMatrixPre.setZero(m_baseZFunctions.cols(), m_baseZFunctions.cols());
			
			m_rhsXFirstTermMatrix.setZero(m_baseXFunctions.cols(), m_baseXFunctions.cols());
			m_rhsYFirstTermMatrix.setZero(m_baseYFunctions.cols(), m_baseYFunctions.cols());
			m_rhsZFirstTermMatrix.setZero(m_baseZFunctions.cols(), m_baseZFunctions.cols());
			
			#pragma omp parallel
			#pragma omp single nowait
			{
			#pragma omp task
				m_rhsXFirstTermMatrixPre = (m_baseXFunctionsTransposed * m_massMatrix * m_baseXFunctions);
			#pragma omp task
				m_rhsYFirstTermMatrixPre = (m_baseYFunctionsTransposed * m_massMatrix * m_baseYFunctions);	
			#pragma omp task
				m_rhsZFirstTermMatrixPre = (m_baseZFunctionsTransposed * m_massMatrix * m_baseZFunctions);	
			}		
		}	
		
		#pragma omp parallel
		#pragma omp single nowait
		{
		#pragma omp task
			m_rhsXFirstTermMatrix = m_rhsXFirstTermMatrixPre * (1. / (m_timeStep * m_timeStep));    //m_rhsFirstTermMatrix = (U.T M U / h^2)
		#pragma omp task
			m_rhsYFirstTermMatrix = m_rhsYFirstTermMatrixPre * (1. / (m_timeStep * m_timeStep));    //m_rhsFirstTermMatrix = (U.T M U / h^2)	
		#pragma omp task
			m_rhsZFirstTermMatrix = m_rhsZFirstTermMatrixPre * (1. / (m_timeStep * m_timeStep));    //m_rhsFirstTermMatrix = (U.T M U / h^2)	
		}
				
		if (m_rhsXFirstTermMatrix.hasNaN() || m_rhsYFirstTermMatrix.hasNaN() || m_rhsZFirstTermMatrix.hasNaN() ) {
			std::cout << "Warning: projected momentum RHS term has NaN values." << std::endl;
		}
		
		std::cout << "Projecting the momentum term LHS matrices to POD subspaces, for the global system ..." << std::endl;
		PDMatrix eps(m_subspaceXLHS_mom.rows(), m_subspaceXLHS_mom.rows());
		eps.setIdentity();
		eps *= 1e-10;
		//std::cout << "Projected the momentum term LHS matrices to POD subspaces, for the global system ..." << std::endl;
		
		// Momentum term: m_subspaceLHS_mom = (U.T M U/h^2)
		if(isPODBasisOrthogonal){  //in the orthogonal case: U.T M U = Identity
			
			m_subspaceXLHS_mom.setIdentity(m_baseXFunctions.cols(), m_baseXFunctions.cols());
			m_subspaceYLHS_mom.setIdentity(m_baseXFunctions.cols(), m_baseXFunctions.cols());
			m_subspaceZLHS_mom.setIdentity(m_baseXFunctions.cols(), m_baseXFunctions.cols());			
			#pragma omp parallel
			#pragma omp single nowait
			{
			#pragma omp task
				m_subspaceXLHS_mom *= (1. / (m_timeStep * m_timeStep)); 
			#pragma omp task
				m_subspaceYLHS_mom *= (1. / (m_timeStep * m_timeStep)); 
			#pragma omp task
				m_subspaceZLHS_mom *= (1. / (m_timeStep * m_timeStep)); 		
			}
		}
		else{
			#pragma omp parallel
			#pragma omp single nowait
			{
			#pragma omp task
				m_subspaceXLHS_mom = m_baseXFunctionsTransposed * m_lhsMatrix * m_baseXFunctions; 
			#pragma omp task
				m_subspaceYLHS_mom = m_baseYFunctionsTransposed * m_lhsMatrix * m_baseYFunctions; 
			#pragma omp task
				m_subspaceZLHS_mom = m_baseZFunctionsTransposed * m_lhsMatrix * m_baseZFunctions; 		
			}
		}
		//std::cout << m_subspaceXLHS_mom.rows() << " " << m_subspaceXLHS_mom.cols() << std::endl;
		//std::cout<< "Projecting momentum terms has been done!.." << std::endl;
		
		if (m_subspaceXLHS_mom.hasNaN() || m_subspaceYLHS_mom.hasNaN() || m_subspaceZLHS_mom.hasNaN() ) {
			std::cout << "Warning: projected momentum LHS term has NaN values." << std::endl;
		}
	}
	
	
	// 2) Compute the constraint part of the global step
	// If using rhs interpolation, let the constraint groups set up the constraint part of the LHS matrix
	if (m_rhsInterpolation) {
		std::cout << "Building and factorizing the complete LHS matrix... " << std::endl;
			
		if (m_usingSkinSubspaces && !m_usePosSnapBases) {    /// Here we have Skinning positionSubspace reduction and rhdInterpolation
			m_subspaceLHS_inner.setZero(m_baseFunctions.cols(), m_baseFunctions.cols());
			m_meshSnapshotsDirectory = m_meshSnapshotsDirectory + "LBS_pos_and_constraint/";

			if (CreateDirectory(m_meshSnapshotsDirectory.c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError())
			{
				std::cout << "Snapshots directory created!: " << m_meshSnapshotsDirectory << std::endl;
			}

			std::cout << "Simulation case: Skinning subspaces for positions and LS Fitting for constraint projection" << std::endl;
			
			// Projection terms for each snapshot group
			for (auto& g : m_snapshotGroups) {
				m_subspaceLHS_inner += g.getLHSMatrixSubspace(m_baseFunctions, m_baseFunctionsTransposed);
				// m_subspaceLHS_inner = Sum_i U.T lambda_i S_i.T S_i U
			}
			// In case of flat bending, there is no bending constraint group but the bending terms
			// still need to be added to the lhs
			if (m_flatBending && !m_bendingConstraints.empty()) {
				for (auto c : m_bendingConstraints) {
					PDMatrix tmp = (m_baseFunctionsTransposed * c->getSelectionMatrixTransposed()) * (c->getSelectionMatrix() * m_baseFunctions);
					tmp *= c->getWeight();
					if (tmp.hasNaN()) {
						std::cout << "Error while constructing lhs..." << std::endl;
					}
					m_subspaceLHS_inner += tmp;
				}
			}
			// Additional constraints: we then add more terms to m_subspaceLHS_inner.
			for (auto c : m_additionalConstraints) {
				PDMatrix tmp = (m_baseFunctionsTransposed * c->getSelectionMatrixTransposed()) * (c->getSelectionMatrix() * m_baseFunctions);
				tmp *= c->getWeight();
				m_subspaceLHS_inner += tmp;
			}
			
			m_lhsMatrixSampled = m_subspaceLHS_mom  + m_subspaceLHS_inner;    // m_lhsMatrixSampled = (1/ h^2) U.T M U + Sum_i U.T lambda_i S_i.T S_i U
			m_denseSolver.compute(m_lhsMatrixSampled);                        // factorizing for the linear global solve
			if (m_denseSolver.info() != Eigen::Success) {
				std::cout << "Warning: Factorization denseSolver of LHS matrix for global system was not successful!.. make sure PROJ_DYN_SPARSIFY is set TRUE!" << std::endl;
			}
		}
		else if(m_usePosSnapBases && !m_usingSkinSubspaces){  /// Here we have POD positionSubspace and rhdInterpolation
			m_subspaceXLHS_inner.setZero(m_baseXFunctions.cols(), m_baseXFunctions.cols());
			m_subspaceYLHS_inner.setZero(m_baseXFunctions.cols(), m_baseXFunctions.cols());
			m_subspaceZLHS_inner.setZero(m_baseXFunctions.cols(), m_baseXFunctions.cols());
			
			m_meshSnapshotsDirectory = m_meshSnapshotsDirectory + "PCA_pos_and_LBS_constraint/";
			if (CreateDirectory(m_meshSnapshotsDirectory.c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError())
			{
				std::cout << "Snapshots directory created!: " << m_meshSnapshotsDirectory << std::endl;
			}
			std::cout << "Simulation case: POD subspaces for positions and LS Fitting for constraint projection" << std::endl;
			
			PDSparseMatrix conMat(m_lhsMatrix.rows(), m_lhsMatrix.cols());
			conMat.setZero();
			std::vector<Eigen::Triplet<PDScalar>> entries;
			for (auto& c : m_constraints) {
				PDSparseMatrixRM& selMat = c->getSelectionMatrix();
				
				for (int k = 0; k < selMat.outerSize(); ++k)
					for (PDSparseMatrixRM::InnerIterator it(selMat, k); it; ++it)
					{
						for (PDSparseMatrixRM::InnerIterator it2(selMat, k); it2; ++it2)
						{
							entries.push_back(Eigen::Triplet<PDScalar>(it.col(), it2.col(), it.value() * it2.value() * c->getWeight()));
						}
					}
			}
			conMat.setFromTriplets(entries.begin(), entries.end());
			//Additional constraints: we then add more terms to m_subspaceLHS_inner.
			for (auto c : m_additionalConstraints) {
				PDSparseMatrix tmp = ( c->getSelectionMatrixTransposed()) * (c->getSelectionMatrix() );
				tmp *= c->getWeight();
				conMat += tmp;
			}
			
			#pragma omp parallel
				#pragma omp single nowait
				{
				#pragma omp task
					m_subspaceXLHS_inner = m_baseXFunctionsTransposed * conMat * m_baseXFunctions; 
				#pragma omp task
					m_subspaceYLHS_inner = m_baseYFunctionsTransposed * conMat * m_baseYFunctions;  
				#pragma omp task

					m_subspaceZLHS_inner = m_baseZFunctionsTransposed * conMat * m_baseZFunctions;  	
				}
				if (m_subspaceXLHS_inner.hasNaN() || m_subspaceYLHS_inner.hasNaN() || m_subspaceZLHS_inner.hasNaN() ) {
					std::cout << "Error: projected constraints LHS term non-orthogonal has NaN values." << std::endl;
				}
			#pragma omp parallel
			#pragma omp single nowait
			{
			#pragma omp task
				m_lhsXMatrixSampled = m_subspaceXLHS_mom + m_subspaceXLHS_inner ;   // U.T (M/h^2 )U  + U.T  lambda S.T S U
			#pragma omp task
				m_lhsYMatrixSampled = m_subspaceYLHS_mom + m_subspaceYLHS_inner ;  
			#pragma omp task
				m_lhsZMatrixSampled = m_subspaceZLHS_mom + m_subspaceZLHS_inner ; 	
			}
			if (m_lhsXMatrixSampled.hasNaN() || m_lhsYMatrixSampled.hasNaN() || m_lhsZMatrixSampled.hasNaN() ) {
				std::cout << "Error: projected LHS has NaN values." << std::endl;
			}
					
			#pragma omp parallel
			#pragma omp single nowait
			{
			#pragma omp task
				m_denseXSolver.compute(m_lhsXMatrixSampled);  
			#pragma omp task
				m_denseYSolver.compute(m_lhsYMatrixSampled);  
			#pragma omp task
				m_denseZSolver.compute(m_lhsZMatrixSampled);  	
			} 
			if (m_denseXSolver.info() != Eigen::Success || m_denseYSolver.info() != Eigen::Success || m_denseZSolver.info() != Eigen::Success) {   
			std::cout << "Warning: Factorization denseSolver X/Y/Z of LHS matrix for global system was not successful!.." << std::endl;
			}
			std::cout << "Factorization denseSolver X/Y/Z of LHS matrix for global system was successful!.." << std::endl;
			//m_subspaceXLHS_inner = m_subspaceLHS_inner.sparseView(0, PROJ_DYN_SPARSITY_CUTOFF);					

		}
		else if (!m_usingPosSubspaces){  // m_rhsInterpolation but no position space reduction

			m_meshSnapshotsDirectory = m_meshSnapshotsDirectory + "LBS_only_constraint/";
			if (CreateDirectory(m_meshSnapshotsDirectory.c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError())
			{
				std::cout << "Snapshots directory created!: " << m_meshSnapshotsDirectory << std::endl;
			}
			std::cout << "Simulation case: No subspace reducgion for positions and LS Fitting for constraint projection" << std::endl;
			
			PDSparseMatrix conMat(m_lhsMatrix.rows(), m_lhsMatrix.cols());

			conMat.setZero();
			std::vector<Eigen::Triplet<PDScalar>> entries;
			for (auto& c : collectedConstraints) {
				PDSparseMatrixRM& selMat = c->getSelectionMatrix();
				for (int k = 0; k < selMat.outerSize(); ++k)
					for (PDSparseMatrixRM::InnerIterator it(selMat, k); it; ++it)
					{
						for (PDSparseMatrixRM::InnerIterator it2(selMat, k); it2; ++it2)
						{
							entries.push_back(Eigen::Triplet<PDScalar>(it.col(), it2.col(), it.value() * it2.value() * c->getWeight()));
						}
					}
			}
			conMat.setFromTriplets(entries.begin(), entries.end());
			//Additional constraints: we then add more terms to m_subspaceLHS_inner.
			for (auto c : m_additionalConstraints) {
				PDSparseMatrix tmp = ( c->getSelectionMatrixTransposed()) * (c->getSelectionMatrix() );
				tmp *= c->getWeight();
				conMat += tmp;
			}
					
			m_lhsMatrix += conMat;         // (M/h^2) + lambda S.T S
			m_lhsMatrix.prune(0, 1e-9f);
			int nnz = m_lhsMatrix.nonZeros();
			
			// Factorize lhs matrix
			StopWatch tmpWatch(10, 10);
			tmpWatch.startStopWatch();
			m_linearFullLHSinterploRHSSolver.analyzePattern(m_lhsMatrix);
			m_linearFullLHSinterploRHSSolver.factorize(m_lhsMatrix);
			tmpWatch.stopStopWatch();
						
			std::cout << "Factorization of the system matrix took " << tmpWatch.lastMeasurement() << " microseconds." << std::endl;
			
			if (m_linearFullLHSinterploRHSSolver.info() != Eigen::Success) {   // I think this case should be included inside "else" above!
			std::cout << "Warning: Factorization denseSolver of LHS matrix for global system was not successful!.. make sure PROJ_DYN_SPARSIFY is set TRUE!" << std::endl;
			}

		}
		
	}
	// In case constraint sampling / rhs interpolation is not used
	// we set up the constraint part of the LHS matrix manually here
	if (!m_rhsInterpolation){
		std::cout << "Building and factorizing the complete LHS matrix... " << std::endl;   
		
		PDSparseMatrix conMat(m_lhsMatrix.rows(), m_lhsMatrix.cols());
		conMat.setZero();
		std::vector<Eigen::Triplet<PDScalar>> entries;
		for (auto& c : m_constraints) {
			PDSparseMatrixRM& selMat = c->getSelectionMatrix();
			for (int k = 0; k < selMat.outerSize(); ++k)
				for (PDSparseMatrixRM::InnerIterator it(selMat, k); it; ++it)
				{
					for (PDSparseMatrixRM::InnerIterator it2(selMat, k); it2; ++it2)
					{
						entries.push_back(Eigen::Triplet<PDScalar>(it.col(), it2.col(), it.value() * it2.value() * c->getWeight()));
					}
				}
		}
			
		conMat.setFromTriplets(entries.begin(), entries.end());
		
		if (m_usingSkinSubspaces&& !m_usePosSnapBases) { // Slow case: using position subspaces but no rhs interpolation
			
			m_meshSnapshotsDirectory = m_meshSnapshotsDirectory + "LBS_only_pos/";
			if (CreateDirectory(m_meshSnapshotsDirectory.c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError())
			{
				std::cout << "Snapshots directory created!: " << m_meshSnapshotsDirectory << std::endl;
			}
			std::cout << "Simulation case: Skinning subspace for positions and no reduction for constraint projection. VERY SLOW " << std::endl;
			// (numeriacally unstable) REQUIRES EXTREMELY SMALL TIME STEP for reasonable visual simulations and becomes even SLOWWWWWWER!!" 
			
			m_subspaceLHS_inner.setZero(m_baseFunctions.cols(), m_baseFunctions.cols());
			
			std::cout << "Constraining it to the subspace..." << std::endl;
			
			m_subspaceLHS_inner = m_baseFunctionsTransposed * conMat * m_baseFunctions;
			m_lhsMatrixSampled = m_subspaceLHS_mom + m_subspaceLHS_inner;

			m_denseSolver.compute(m_lhsMatrixSampled);  // 
			std::cout << "Size of sampled, dense lhs mat: " << m_lhsMatrixSampled.rows() << ", " << m_lhsMatrixSampled.cols() << std::endl;
		}		
		else if (m_usePosSnapBases && !m_usingSkinSubspaces) { // Slow case: using position subspaces but no rhs interpolation
			
			if(m_usingQDEIMComponents){
				if(m_solveDeimLS){
					m_meshSnapshotsDirectory = m_meshSnapshotsDirectory + "PCA_pos_and_LSDEIM_constraint/";
					
					std::cout << "Simulation case: POD subspace for positions and DEIM/QDEIM for constraint projection, using Least square" << std::endl;
				}
				else{
					m_meshSnapshotsDirectory = m_meshSnapshotsDirectory + "PCA_pos_and_DEIM_constraint/";
					std::cout << "Simulation case: POD subspace for positions and DEIM/QDEIM for constraint projection" << std::endl;
				}
			}
			else
			{
				std::cout << "Simulation case: POD subspace for positions and no reduction for constraint projection" << std::endl;
				m_meshSnapshotsDirectory = m_meshSnapshotsDirectory + "PCA_only_pos/";
			}
			if (CreateDirectory(m_meshSnapshotsDirectory.c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError())
			{
				std::cout << "Snapshots directory created!: " << m_meshSnapshotsDirectory << std::endl;
			}

			m_subspaceXLHS_inner.setZero(m_baseXFunctions.cols(), m_baseXFunctions.cols());
			m_subspaceYLHS_inner.setZero(m_baseYFunctions.cols(), m_baseYFunctions.cols());
			m_subspaceZLHS_inner.setZero(m_baseZFunctions.cols(), m_baseZFunctions.cols());

			
			std::cout << "Projecting the LHS matrices to the subspace..." << std::endl;
			
			if(isPODBasisOrthogonal){  //in the orthogonal case: U.T M^-1 Sum lambda S.T S M^-1 U
				#pragma omp parallel
				#pragma omp single nowait
				{
				#pragma omp task
					m_subspaceXLHS_inner = m_baseXFunctionsTransposed * conMat * m_baseXFunctions * m_massMatrixInv; 
				#pragma omp task
					m_subspaceYLHS_inner = m_baseYFunctionsTransposed * conMat * m_baseYFunctions * m_massMatrixInv;  
				#pragma omp task
					m_subspaceZLHS_inner = m_baseZFunctionsTransposed * conMat * m_baseZFunctions * m_massMatrixInv;  	
				}
				if (m_subspaceXLHS_inner.hasNaN() || m_subspaceYLHS_inner.hasNaN() || m_subspaceZLHS_inner.hasNaN() ) {
					std::cout << "Error: projected constraints LHS term orthogonal has NaN values." << std::endl;
				}
			}
			else{	 //in the non-orthogonal case: U.T Sum lambda S.T S U
				#pragma omp parallel
				#pragma omp single nowait
				{
				#pragma omp task
					m_subspaceXLHS_inner = m_baseXFunctionsTransposed * conMat * m_baseXFunctions; 
				#pragma omp task
					m_subspaceYLHS_inner = m_baseYFunctionsTransposed * conMat * m_baseYFunctions;  
				#pragma omp task
					m_subspaceZLHS_inner = m_baseZFunctionsTransposed * conMat * m_baseZFunctions;  	
				}
				if (m_subspaceXLHS_inner.hasNaN() || m_subspaceYLHS_inner.hasNaN() || m_subspaceZLHS_inner.hasNaN() ) {
					std::cout << "Error: projected constraints LHS term non-orthogonal has NaN values." << std::endl;
				}
			}
			
			#pragma omp parallel
			#pragma omp single nowait
			{
			#pragma omp task
				m_lhsXMatrixSampled = m_subspaceXLHS_mom + m_subspaceXLHS_inner ; 
			#pragma omp task
				m_lhsYMatrixSampled = m_subspaceYLHS_mom + m_subspaceYLHS_inner ;  
			#pragma omp task
				m_lhsZMatrixSampled = m_subspaceZLHS_mom + m_subspaceZLHS_inner ; 	
			}
			if (m_lhsXMatrixSampled.hasNaN() || m_lhsYMatrixSampled.hasNaN() || m_lhsZMatrixSampled.hasNaN() ) {
				std::cout << "Error: projected LHS has NaN values." << std::endl;
			}
					
			#pragma omp parallel
			#pragma omp single nowait
			{
			#pragma omp task
				m_denseXSolver.compute(m_lhsXMatrixSampled);  
			#pragma omp task
				m_denseYSolver.compute(m_lhsYMatrixSampled);  
			#pragma omp task
				m_denseZSolver.compute(m_lhsZMatrixSampled);  	
			} 
			if (m_denseXSolver.info() != Eigen::Success || m_denseYSolver.info() != Eigen::Success || m_denseZSolver.info() != Eigen::Success) {   
			std::cout << "Warning: Factorization denseSolver X/Y/Z of LHS matrix for global system was not successful!.." << std::endl;
			}	
		}
		else { // Full simulation: here neither position space nor constraint projection reduction ===> no reduction at all!!
			m_meshSnapshotsDirectory = m_meshSnapshotsDirectory + "FOM/";
			if (CreateDirectory(m_meshSnapshotsDirectory.c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError())
			{
				std::cout << "Snapshots directory created!: " << m_meshSnapshotsDirectory << std::endl;
			}
			std::cout << "Simulation case: No REDUCTION: we run FullSpace positions and FullSpace constraint projection" << std::endl;

			m_lhsMatrix += conMat;
			m_lhsMatrix.prune(0, 1e-9f);
			int nnz = m_lhsMatrix.nonZeros();
			
			// Factorize lhs matrix
			StopWatch tmpWatch(10, 10);
			tmpWatch.startStopWatch();
			m_linearSolver.analyzePattern(m_lhsMatrix);
			m_linearSolver.factorize(m_lhsMatrix);
			tmpWatch.stopStopWatch();
						
			std::cout << "Factorization of the system matrix took " << tmpWatch.lastMeasurement() << " microseconds." << std::endl;
			
			if (m_linearSolver.info() != Eigen::Success) {   // I think this case should be included inside "else" above!
			std::cout << "Warning: Factorization denseSolver of LHS matrix for global system was not successful!.. make sure PROJ_DYN_SPARSIFY is set TRUE!" << std::endl;
			}
		}
		
		
		// when rhsInterpolation
		// After the lhs has been constructed, if flat bending is desired,
		// the bending constraints can now be thrown away! (?)
		if (m_flatBending) {
			for (ProjDynConstraint* c : m_bendingConstraints) {
				auto const& bc = std::find(m_constraints.begin(), m_constraints.end(), c);
				if (bc != m_constraints.end()) {
					m_constraints.erase(bc);
				}
			}
		}
	}
	
	
	m_recomputeFactorization = false;

	// Now, all sampled constraints should have been added and the used vertices can be updated,
	// and the constraints can be updated to use this list
	
	/* note: rhsInterpolation uses a subset of the vertices used by skinning subspaces:
	--> Therefore, in case rhsInterpolation we just updateUsedVertices,
	   		in case m_usingSkinSubspaces we need to add the rest of the vertices from m_samples then updateUsedVertices,
	   		while, otherwise in case of full simulations or POD subspaces we need all verties.
	   		TODO: consider a list of vertices when using splocs zum beispiel!
	*/
	if (m_rhsInterpolation && m_usingSkinSubspaces && !m_usePosSnapBases) {
		//std::cout << "Determining used vertices..." << std::endl;
		updateUsedVertices();
	}
	else if (m_usingSkinSubspaces && !m_usePosSnapBases) {
		for (unsigned int v : m_samples) {
			m_additionalUsedVertices.push_back(v);
		}
		updateUsedVertices();
	}
	else if (m_rhsInterpolation && m_usePosSnapBases){
		
		for (unsigned int v : m_constraintVertexSamples) { // here we use only constraint samples
			m_additionalUsedVertices.push_back(v);
		}
		/*
		for (unsigned int v = 0; v < m_numVertices; v++) {
			m_additionalUsedVertices.push_back(v);
		}*/
		updateUsedVertices();
		
	}
	else{                                              
		m_usedVertices.clear();
		for (unsigned int v = 0; v < m_numVertices; v++) {
			m_usedVertices.push_back(v);
		}
	}
	//std::cout <<m_usedVertices.size() << std::endl;
	
	// Optional sparsification of matrices
	if (m_usingPosSubspaces && m_useSparseMatricesForSubspace) {
		//std::cout << "PROJ_DYN_SPARSIFY is set TRUE.. Sparsifying the LHS complete matrix..." << std::endl;
		
		if(m_usingSkinSubspaces && !m_usePosSnapBases){
			PDSparseMatrix lhsMatrixSampledSparse = m_lhsMatrixSampled.sparseView(0, PROJ_DYN_SPARSITY_CUTOFF_HIGH_PREC);
			m_subspaceSystemSolverSparse.compute(lhsMatrixSampledSparse);
			if (m_subspaceSystemSolverSparse.info() != Eigen::Success) {
				std::cout << "Warning: Factorization of the sparse LHS matrix for the global step was not successful!" << std::endl;
				PDSparseMatrix eps(lhsMatrixSampledSparse.rows(), lhsMatrixSampledSparse.rows());
				eps.setIdentity();
				eps *= 1e-12;
				while (m_subspaceSystemSolverSparse.info() != Eigen::Success && eps.coeff(0, 0) < 1e-10) {
					std::cout << "Adding small diagonal entries (" << eps.coeff(0, 0) << ")..." << std::endl;
					lhsMatrixSampledSparse += eps;
					eps *= 2;
					m_subspaceSystemSolverSparse.compute(eps);
				}
			}
			else{
				std::cout << "Factorization of the sparse LHS matrix for the global step was successful!" << std::endl;
			}
			m_rhsFirstTermMatrixSparse = m_rhsFirstTermMatrix.sparseView(0, PROJ_DYN_SPARSITY_CUTOFF_HIGH_PREC);
			updateParallelVUpdateBlocks();
		
		}
		if(m_usePosSnapBases && !m_usingSkinSubspaces){
			PDSparseMatrix lhsXMatrixSampledSparse = m_lhsXMatrixSampled.sparseView(0, PROJ_DYN_SPARSITY_CUTOFF_HIGH_PREC);	
			PDSparseMatrix lhsYMatrixSampledSparse = m_lhsYMatrixSampled.sparseView(0, PROJ_DYN_SPARSITY_CUTOFF_HIGH_PREC);
			PDSparseMatrix lhsZMatrixSampledSparse = m_lhsZMatrixSampled.sparseView(0, PROJ_DYN_SPARSITY_CUTOFF_HIGH_PREC);
					
			#pragma omp parallel
			#pragma omp single nowait
			{
			#pragma omp task
				m_subspaceXSystemSolverSparse.compute(lhsXMatrixSampledSparse);
			#pragma omp task
				m_subspaceYSystemSolverSparse.compute(lhsYMatrixSampledSparse);
			#pragma omp task
				m_subspaceZSystemSolverSparse.compute(lhsZMatrixSampledSparse);
			}
			
			if (m_subspaceXSystemSolverSparse.info() != Eigen::Success) {
				std::cout << "Warning: Factorization of the sparse X LHS matrix for the global step was not successful!" << std::endl;
				PDSparseMatrix eps(lhsXMatrixSampledSparse.rows(), lhsXMatrixSampledSparse.rows());
				eps.setIdentity();
				eps *= 1e-12;
				while (m_subspaceXSystemSolverSparse.info() != Eigen::Success && eps.coeff(0, 0) < 1e-10) {
					std::cout << "Adding small diagonal entries (" << eps.coeff(0, 0) << ")..." << std::endl;
					lhsXMatrixSampledSparse += eps;
					eps *= 2;
					m_subspaceXSystemSolverSparse.compute(lhsXMatrixSampledSparse);
				}
			}
			else if (m_subspaceYSystemSolverSparse.info() != Eigen::Success) {
				std::cout << "Warning: Factorization of the sparse Y LHS matrix for the global step was not successful!" << std::endl;
				PDSparseMatrix eps(lhsYMatrixSampledSparse.rows(), lhsYMatrixSampledSparse.rows());
				eps.setIdentity();
				eps *= 1e-12;
				while (m_subspaceYSystemSolverSparse.info() != Eigen::Success && eps.coeff(0, 0) < 1e-10) {
					std::cout << "Adding small diagonal entries (" << eps.coeff(0, 0) << ")..." << std::endl;
					lhsYMatrixSampledSparse += eps;
					eps *= 2;
					m_subspaceYSystemSolverSparse.compute(lhsYMatrixSampledSparse);
				}
			}
			else if (m_subspaceZSystemSolverSparse.info() != Eigen::Success) {
				std::cout << "Warning: Factorization of the sparse Z LHS matrix for the global step was not successful!" << std::endl;
				PDSparseMatrix eps(lhsZMatrixSampledSparse.rows(), lhsZMatrixSampledSparse.rows());
				eps.setIdentity();
				eps *= 1e-12;
				while (m_subspaceZSystemSolverSparse.info() != Eigen::Success && eps.coeff(0, 0) < 1e-10) {
					std::cout << "Adding small diagonal entries (" << eps.coeff(0, 0) << ")..." << std::endl;
					lhsZMatrixSampledSparse += eps;
					eps *= 2;
					m_subspaceZSystemSolverSparse.compute(lhsZMatrixSampledSparse);
				}
			
			}			
			else{
				std::cout << "Factorization of the sparse LHS matrix for the global step was successful!" << std::endl;
			}
			m_rhsXFirstTermMatrixSparse = m_rhsXFirstTermMatrix.sparseView(0, PROJ_DYN_SPARSITY_CUTOFF_HIGH_PREC);  // Only momentum term of RHS
			m_rhsYFirstTermMatrixSparse = m_rhsYFirstTermMatrix.sparseView(0, PROJ_DYN_SPARSITY_CUTOFF_HIGH_PREC);
			m_rhsZFirstTermMatrixSparse = m_rhsZFirstTermMatrix.sparseView(0, PROJ_DYN_SPARSITY_CUTOFF_HIGH_PREC);
			std::cout << "First RHS term has been also sparsified.." << std::endl;
			//updateParallelVUpdateBlocks();/// TODO :make sure this update is aware of the pod case first before using it
		
		}
		
	}
	
	m_collidedVerts = new bool[m_numVertices];
	
	for (int i = 0; i < m_numVertices; i++) m_collidedVerts[i] = false;

	m_isSetup = true;
	m_collisionCorrection = false;
	//m_planeBounceCorrection = false;
	m_grippedVertices.clear();

	m_initialPos = m_positions;
	m_initialPosSub = m_positionsSubspace;

	// Precompute the weighted external forces that will appear on the rhs
	recomputeWeightedForces();

	m_precomputationStopWatch.stopStopWatch();
	
	

	if(m_usingQDEIMComponents && !m_solveDeimLS){	// TODO: parallelize!

		/*
		UTSTx.resize(usedConstraints->size()), UTSTy.resize(usedConstraints->size()), UTSTz.resize(usedConstraints->size());
		for (int ind = 0; ind < usedConstraints->size(); ind++) {  // usedConstraints->size() = numConstraints * ConstraintsSize ==e*p
			
			PDSparseMatrix ST;    // selection.T matrices
			ST = usedConstraints->at(ind)->getSelectionMatrixTransposed();   // (numVertices, p)
			
			// e-times
			UTSTx[ind] = m_baseXFunctionsTransposedSparse * ST;    // each size (r_pod , p)
			UTSTy[ind] = m_baseYFunctionsTransposedSparse * ST;
			UTSTz[ind] = m_baseZFunctionsTransposedSparse * ST;
			
			//std::cout << ST.innerSize() << " " << ST.outerSize()<< " " << usedConstraints->size() << std::endl;
		
			p_constaintSize = ST.outerSize();
			
		}
		
		*/
		int p = 1;   // constraint size
		if (m_TetStrainOnly) {
			p = 3;
		}
		std::cout << "Preparing QDEIM RHS matrices ...";
		UTSTMx.setZero(m_numPosPODModes+1, deimBasisCols), UTSTMy.setZero(m_numPosPODModes+1, deimBasisCols), UTSTMz.setZero(m_numPosPODModes+1, deimBasisCols);
		STMx.setZero(m_numVertices, deimBasisCols), STMy.setZero(m_numVertices, deimBasisCols), STMz.setZero(m_numVertices, deimBasisCols);
		for(int ind = 0; ind < usedConstraints->size(); ind++){
			//int p = p_constaintSize;
			PDMatrix bx, by, bz; 
			bx = m_xMqdeim.block(ind* p , 0, p, deimBasisCols);  // (p, deimBasisCols) // TODO: sparcify M (?)
			by = m_yMqdeim.block(ind* p , 0, p, deimBasisCols);
			bz = m_zMqdeim.block(ind* p , 0, p, deimBasisCols);
			
			PDSparseMatrix ST_ind = usedConstraints->at(ind)->getSelectionMatrixTransposed();   // (numVertices, p)
			PDScalar weight_ind = usedConstraints->at(ind)->getWeight();
			for (int k=0; k < deimBasisCols; k++){
				/*
				fastDensePlusSparseTimesDenseMat(UTSTMx, UTSTx[ind] , bx, k);  // size (r_pod , kp) dense matrix
				fastDensePlusSparseTimesDenseMat(UTSTMy, UTSTy[ind] , by, k);
				fastDensePlusSparseTimesDenseMat(UTSTMz, UTSTz[ind] , bz, k); 

				*/
				
				fastDensePlusSparseTimesDenseMat(STMx, ST_ind, bx, k, weight_ind); // (n, deimBasisCols)
				fastDensePlusSparseTimesDenseMat(STMy, ST_ind, by, k, weight_ind); 
				fastDensePlusSparseTimesDenseMat(STMz, ST_ind, bz, k, weight_ind); 
			}
		}

		UTSTMx = m_baseXFunctionsTransposedSparse * STMx ;    // (r_pod, deimBasisCols)
		UTSTMy = m_baseYFunctionsTransposedSparse * STMy ;
		UTSTMz = m_baseZFunctionsTransposedSparse * STMz ;

		std::cout << " done." << std::endl;
	}	
				
#ifndef EIGEN_DONT_PARALLELIZE
	Eigen::setNbThreads(PROJ_DYN_EIGEN_NUM_THREADS);
#endif
//std::cout << m_velocitiesSubspace << std::endl;
}


void ProjDynSimulator::resetPositions() {
	m_positions = m_initialPos;
	m_positionsSubspace = m_initialPosSub;
	m_velocities.setZero();
	m_velocitiesSubspace.setZero();
	m_collisionCorrection = false;
	//m_planeBounceCorrection = false;
	m_positionCorrectionsUsedVs.setZero();
	releaseGrip();
}
/*
void PD::ProjDynSimulator::resetVelocities()
{
	std::cout << "===ProjDynSimulator::resetVelocities===" << std::endl;
	m_velocities.setZero();
	m_velocitiesSubspace.setZero();
	m_velocitiesUsedVs.setZero();
}
*/

/* Specify a list of vertices who should be constrained to temporarily fixed positions.
vInds should be a list of vertex indices into the vertex positions matrix, while
gripPos is a k by 3 matrix, where k is the number of gripped vertices, containing
the positions of the gripped vertices */
void PD::ProjDynSimulator::setGrip(std::vector<unsigned int>& vInds, PDPositions gripPos)
{
	m_grippedVertices = vInds;
	m_grippedVertexPos = gripPos;
}

/* List of all used vertices, i.e. vertices that appear in any sampled constraint
projection. Provides a map via usedVertices[i_sub] = i_full, where the indices
i_sub are used, for example, in set grip, and i_full are the corresponding vertices
of the full mesh.*/
/*
std::vector<unsigned int>& PD::ProjDynSimulator::getUsedVertices()
{
	std::cout << "===PD::ProjDynSimulator::getUsedVertices===" << std::endl;
	return m_usedVertices;
}
*/
void PD::ProjDynSimulator::releaseGrip()
{
	m_grippedVertices.clear();
}


void ProjDynSimulator::step(int numIterations)
{
	
	/*
	// Uncomment to track energy
	std::ofstream StoreEnergyFile;   // file to track energy
        StoreEnergyFile.open ("recordeEnergy.txt" , std::ofstream::app);
	//StoreEnergyFile << "numIterations" << "," << "frameCount" << "," << "totalEnergie" << std::endl;
	*/
	
	if (m_numIterations < 0) m_numIterations = numIterations;

	m_totalStopWatch.startStopWatch();
	m_surroundingBlockStopWatch.startStopWatch();

	if (!m_isSetup) {
		std::cout << "Constraints or external forces have changed since last setup call, therefore setup() will be called now!" << std::endl;
		setup();
	}

	PDPositions s;     // s = q(t) + h v(t) + h^2 M^(-1) f_{ext}
	PDPositions oldPos, oldFullPos;
	
	//************************************
	// Compute s and handle collisions/user interaction
	//************************************
	
	// Compute s and handle collisions/user interaction case of m_usingSkinSubspaces with/out m_rhsInterpolation
	if(m_usingSkinSubspaces && !m_usePosSnapBases){
		if (!m_rhsInterpolation) {
			oldFullPos = m_positions;
		}
		// Store previous positions for velocitiy computation at the end of the timestep
		oldPos = m_positionsSubspace;
		
		
		// If there has been a collision in the last step, handle repulsion and friction:
		if (m_collisionCorrection) { 
			
			if(m_rhsInterpolation){
				// Get actual velocities for used vertices
				updatePositionsSampling(m_velocitiesUsedVs, m_velocitiesSubspace, true);
				// Remove tangential movement and add repulsion movement from collided vertices
				PROJ_DYN_PARALLEL_FOR
					for (int v = 0; v < m_velocitiesUsedVs.rows(); v++) {
						if (m_positionCorrectionsUsedVs.row(v).norm() > 1e-12) {
							PDVector tangentialV = m_velocitiesUsedVs.row(v) - m_velocitiesUsedVs.row(v).dot(m_positionCorrectionsUsedVs.row(v)) * m_positionCorrectionsUsedVs.row(v);
							tangentialV *= (1. - m_frictionCoeff);
							tangentialV += m_positionCorrectionsUsedVs.row(v) * m_repulsionCoeff;
							m_velocitiesUsedVs.row(v) = tangentialV;
						}
					}
				// Project full velocities back to subspace via interpolation of the velocities on used vertices
			
				PROJ_DYN_PARALLEL_FOR
					for (int d = 0; d < 3; d++) {
						if (m_useSparseMatricesForSubspace) {
							m_velocitiesSubspace.col(d) = m_usedVertexInterpolatorSparse.solve(m_usedVertexInterpolatorRHSMatrixSparse * m_velocitiesUsedVs.col(d));
						}
						else {
							m_velocitiesSubspace.col(d) = m_usedVertexInterpolator.solve(m_usedVertexInterpolatorRHSMatrix * m_velocitiesUsedVs.col(d));
						}
					}
			}
			else{    // not rhsInterpolation
				//TODO: make sure that used vertices here means from m_samples
				//updatePositionsSampling(m_velocitiesUsedVs, m_velocitiesSubspace, false);   
				
						
				// Remove tangential movement and add repulsion movement from collided vertices
				PROJ_DYN_PARALLEL_FOR
					for (int v = 0; v < m_numVertices; v++) {
						if (m_positionCorrections.row(v).norm() > 1e-12) {
							PDVector tangentialV = m_velocities.row(v) - m_velocities.row(v).dot(m_positionCorrections.row(v)) * m_positionCorrections.row(v);
							tangentialV *= (1. - m_frictionCoeff);
							tangentialV += m_positionCorrections.row(v) * m_repulsionCoeff;
							m_velocities.row(v) = tangentialV;
						}
					}
				/*	
				// Project full velocities back to subspace via interpolation of the velocities on used vertices: 
				// note in this case we use the solvers because basis is noy assumed orthogonal
				PROJ_DYN_PARALLEL_FOR
					for (int d = 0; d < 3; d++) {
						if (m_useSparseMatricesForSubspace) {
							m_velocitiesSubspace.col(d) = m_usedVertexInterpolatorSparse.solve(m_usedVertexInterpolatorRHSMatrixSparse * m_velocities.col(d));
						}
						else {
							m_velocitiesSubspace.col(d) = m_usedVertexInterpolator.solve(m_usedVertexInterpolatorRHSMatrix * m_velocities.col(d));
							
						}
					}
					
					std::cout << "Error! .. solvers for vertices interpolation not working, we need to use another for non-positiveSemiDefinit matrices" << std::endl;  */
					
				projectToSubspace(m_velocitiesSubspace, m_velocities, false);
					
			}
			//std::cout << "handled collision" << std::endl;
		}
		
		PDScalar blowupFac = (10. - m_blowupStrength) / 9.;
		// Compute s:  //note: s = q(t) + h v(t) + h^2 M^(-1) f_{ext}, here q and v are in the subspace
		s = m_positionsSubspace + m_timeStep * m_velocitiesSubspace + m_fExtWeightedSubspace + blowupFac * m_fGravWeightedSubspace;
		if (m_rayleighDampingAlpha > 0) {
			s = s - m_timeStep * m_rayleighDampingAlpha * m_velocitiesSubspace;
		}

		// compute and store energy/objective before initializing subspositions with s
		//PDScalar energie = evaluateEnergy(m_positionsSubspace,  s);
		//StoreEnergyFile << numIterations<< "," << m_frameCount << "," << energie << std::endl;
		 
		// s is also the initial guess for the updated sub/positions
		m_positionsSubspace = s;
		
		// Compute vertex positions on vertices involved in the computation of sampled constraints
		updatePositionsSampling(m_positionsUsedVs, m_positionsSubspace, true);
		
		
		// Correct vertex positions of gripped and collided vertices
		handleGripAndCollisionsUsedVs(s, true);       // true means to update the vertices after handling collection 
	}
	// Compute s and handle collisions/user interaction case of m_usePosSnapBases similaly with/out m_rhsInterpolation	
	else if (m_usePosSnapBases && !m_usingSkinSubspaces) {
		
		if (!m_rhsInterpolation ) {
			oldFullPos = m_positions;
			
		}

		// Store previous positions for velocitiy computation at the end of the timestep
		oldPos = m_positionsSubspace;
		
		// If there has been a collision in the last step, handle repulsion and friction:
		if (m_collisionCorrection) { 

			if (m_rhsInterpolation){
				
				//std::cout << m_velocitiesUsedVs.rows() << std::endl;
				// Get actual velocities 
				updatePODPositionsSampling(m_velocitiesUsedVs, m_velocitiesSubspace, true);    // podUsedVerticesOnly = true
				if (m_velocities.hasNaN() ) {
					std::cout << "Warning: updated fullVelocities has NaN values." << std::endl;
				}
				//std::cout << m_positionCorrections << std::endl;
				
				// Remove tangential movement and add repulsion movement from collided vertices
				// this is what makes the mesh notices the collisions at the floor, otherwise it just keep faling
				PROJ_DYN_PARALLEL_FOR
					for (int v = 0; v < m_velocitiesUsedVs.rows(); v++) {
						if (m_positionCorrectionsUsedVs.row(v).norm() > 1e-12) {
							PDVector tangentialV = m_velocitiesUsedVs.row(v) - m_velocitiesUsedVs.row(v).dot(m_positionCorrectionsUsedVs.row(v)) * m_positionCorrectionsUsedVs.row(v);
							tangentialV *= (1. - m_frictionCoeff);
							tangentialV += m_positionCorrectionsUsedVs.row(v) * m_repulsionCoeff;
							m_velocitiesUsedVs.row(v) = tangentialV;
						}
					}
				// Project full velocities back to subspace via interpolation of the velocities on used vertices
			
				
					if (m_useSparseMatricesForSubspace) {
						#pragma omp parallel
						#pragma omp single nowait
						{
						#pragma omp task
							m_velocitiesSubspace.col(0) = m_usedVertexXInterpolatorSparse.solve(m_usedVertexXInterpolatorRHSMatrixSparse * m_velocitiesUsedVs.col(0));
						#pragma omp task
							m_velocitiesSubspace.col(1) = m_usedVertexYInterpolatorSparse.solve(m_usedVertexYInterpolatorRHSMatrixSparse * m_velocitiesUsedVs.col(1));
						#pragma omp task
							m_velocitiesSubspace.col(2) = m_usedVertexZInterpolatorSparse.solve(m_usedVertexZInterpolatorRHSMatrixSparse * m_velocitiesUsedVs.col(2));
						}					
					}
					else {
						m_velocitiesSubspace.col(0) = m_usedVertexXInterpolator.solve(m_usedVertexXInterpolatorRHSMatrix * m_velocitiesUsedVs.col(0));
						m_velocitiesSubspace.col(1) = m_usedVertexYInterpolator.solve(m_usedVertexYInterpolatorRHSMatrix * m_velocitiesUsedVs.col(1));
						m_velocitiesSubspace.col(2) = m_usedVertexZInterpolator.solve(m_usedVertexZInterpolatorRHSMatrix * m_velocitiesUsedVs.col(2));
					}
				
			}
			else{
				// Get actual velocities for used vertices
				updatePODPositionsSampling(m_velocities, m_velocitiesSubspace, podUsedVerticesOnly);
				//std::cout << m_velocities.maxCoeff() << std::endl;
				//std::cout << m_positionCorrections.maxCoeff() << std::endl;    
				if (m_velocities.hasNaN() ) {
					std::cout << "Warning: updated fullVelocities has NaN values." << std::endl;
				}
				//std::cout << m_positionCorrections << std::endl;
				
				// Remove tangential movement and add repulsion movement from collided vertices
				// this is what makes the mesh notices the collisions at the floor, otherwise it just keep faling
				PROJ_DYN_PARALLEL_FOR
						for (int v = 0; v < m_velocities.rows(); v++) {
							if (m_positionCorrections.row(v).norm() > 1e-12) {
							
								PDVector tangentialV = m_velocities.row(v) - m_velocities.row(v).dot(m_positionCorrections.row(v)) * m_positionCorrections.row(v);
								tangentialV *= (1. - m_frictionCoeff);
								tangentialV += m_positionCorrections.row(v) * m_repulsionCoeff;
								m_velocities.row(v) = tangentialV;
							}
						}
				
				// Project full velocities back to subspace after they were updated due to collision
				if (m_useSparseMatricesForSubspace) {
					projectToSparsePODSubspace(m_velocitiesSubspace, m_velocities, isPODBasisOrthogonal);	
				}
				else {
					projectToPODSubspace(m_velocitiesSubspace, m_velocities, isPODBasisOrthogonal);
				}
			
			}
			//std::cout << "handled collision" << std::endl;
		}
		
		if (m_positionsSubspace.hasNaN() ) {
					std::cout << "Error: computed m_positionsSubspace has NaN values." << std::endl;
		}
		if (m_velocitiesSubspace.hasNaN() ) {
					std::cout << "Error: computed m_velocitiesSubspace has NaN values." << std::endl;
		}
		if (m_fExtWeightedSubspace.hasNaN() ) {
					std::cout << "Error: computed m_fExtWeightedSubspace has NaN values." << std::endl;
		}
		if (m_fGravWeightedSubspace.hasNaN() ) {
					std::cout << "Error: computed m_fGravWeightedSubspace has NaN values." << std::endl;
		}
		
		// Compute s:  //note: s = q(t) + h v(t) + h^2 M^(-1) f_{ext}, here q and v are in the subspace
		PDScalar blowupFac = (10. - m_blowupStrength) / 9.;
		s = m_positionsSubspace + m_timeStep * m_velocitiesSubspace + m_fExtWeightedSubspace + blowupFac * m_fGravWeightedSubspace;
		if (m_rayleighDampingAlpha > 0) {
			s = s - m_timeStep * m_rayleighDampingAlpha * m_velocitiesSubspace;
		}
		if (s.hasNaN() ) {
					std::cout << "Error: computed s has NaN values." << std::endl;
		}
		 
		// s is also the initial guess for the updated sub/positions
		m_positionsSubspace = s;
					
		// fullPos = Ubasis * subPos
		if (m_rhsInterpolation) updatePODPositionsSampling(m_positionsUsedVs, m_positionsSubspace, true);
		else                    updatePODPositionsSampling(m_positions, m_positionsSubspace, false);
		
		if(m_rhsInterpolation) handleGripAndCollisionsUsedVs(s, true);
		else handleGripAndCollisionsUsedVs(s, false);
		/*	
		// TODO: Change position of gripped vertices
		if (m_grippedVertices.size() > 0) {
			for (unsigned int i = 0; i < m_grippedVertices.size(); i++) {
				s.row(m_grippedVertices[i]) = m_grippedVertexPos.row(i);
			}
		} */
		/*
		if(m_useSparseMatricesForSubspace){
			projectToSparsePODSubspace(m_positionsSubspace , m_positions, isPODBasisOrthogonal);
		}
		else{
			projectToPODSubspace(m_positionsSubspace , m_positions, isPODBasisOrthogonal);
		}
		*/
		//updatePODPositionsSampling(m_positions, m_positionsSubspace, podUsedVerticesOnly);
		
		if (m_positions.hasNaN() ) {
					std::cout << "Error: computed m_positions has NaN values." << std::endl;
		}

		
	}
	//}
		
	else { // Simplified computations for full position simulations  (also used when we have no pos reduction but rhsInterpolation!)
	
		if(recordingPSnapshots){
		/* when recording nonlinear constaints projections p(positionsSnapshots)
		   we read the stored position snapshots (no obvious difference from using the current frame)*/
			Eigen::MatrixXd vertsTemp;
			Eigen::MatrixXi facesTemp;
			
			if(igl::readOFF("/home/shaimaa/libigl/tutorial/doubleHRPD/meshFrames_fullSimulationsOFF/bunny/fullPositionsFewFrames/m_bunnyPositions_"+ std::to_string(m_frameCount)+ ".off", vertsTemp, facesTemp)){
				// here we convert to PD types:
				if(vertsTemp.cols() == 0) std::cout << " no positions were found...." << std::endl;
				m_positions = PD::PDPositions(vertsTemp);
				oldPos = m_positions;
				std::cout << "-------------------------------------------------------" <<std::endl;
				
			}
		}
		else{
			oldPos = m_positions;
		}
		PDScalar blowupFac = (10. - m_blowupStrength) / 9.;
		
		// Compute s:  //note: s = q(t) + h v(t) + h^2 M^(-1) f_{ext}, and here q and v are in the full space
		s = m_positions + m_timeStep * m_velocities + m_fExtWeighted + blowupFac * m_fGravWeighted;
		if (m_rayleighDampingAlpha > 0) {
			s = s - m_timeStep * m_rayleighDampingAlpha * m_velocities;
		}
		PROJ_DYN_PARALLEL_FOR
			for (int v = 0; v < m_positions.rows(); v++) {
				resolveCollision(v, s, m_positionCorrections);
			} 
			
		// Change position of gripped vertices
		if (m_grippedVertices.size() > 0) {
			for (unsigned int i = 0; i < m_grippedVertices.size(); i++) {
				s.row(m_grippedVertices[i]) = m_grippedVertexPos.row(i);
			}
		}
		
		// compute energy/objective 
		//PDScalar energie = evaluateEnergy(m_positions,  s);
		//StoreEnergyFile << numIterations << ","<< m_frameCount << "," << energie << std::endl;
		
		
		// initialize full positions
		m_positions = s;
		
	}
		
	// Some additional initializations/updates
	if (m_usingSkinSubspaces && m_constraintSamplesChanged && !m_usePosSnapBases) {
		updateUsedVertices();
	
		updatePositionsSampling(m_positionsUsedVs, m_positionsSubspace, true);     // update at used vertices only = true
	
	}
	if (!m_usingSkinSubspaces && m_constraintSamplesChanged && m_usePosSnapBases && m_rhsInterpolation) {
	
		updateUsedVertices();

	
		updatePODPositionsSampling(m_positionsUsedVs, m_positionsSubspace, true);     // update at used vertices only = true
	
	}
		
	std::vector< ProjDynConstraint* >* usedConstraints = &m_constraints;
	if (m_rhsInterpolation || (m_usingQDEIMComponents && m_solveDeimLS) ){
	
		usedConstraints = &m_sampledConstraints;
	}
	int numConstraints = usedConstraints->size();
	
	std::vector< PDPositions > currentAuxilaries(numConstraints);
	
	int c_chunk = numConstraints / (omp_get_num_threads() * 100);
	int v_chunk = m_numVertices / (omp_get_num_threads() * 100);
	m_surroundingBlockStopWatch.stopStopWatch();

	//************************************
	// Local global loop
	//************************************
	for (int i = 0; i < numIterations; i++) { 

		//************************************
		// Local step: Constraint projections
		//************************************
		m_localStepStopWatch.startStopWatch();
		
		// Start by computing the local term: 
		if (m_rhsInterpolation) { // Local step approximation via fitting method with/out pos reduction
			m_localStepOnlyProjectStopWatch.startStopWatch();
			
			if (m_usingSkinSubspaces && !m_usePosSnapBases) { // here rhs2= lambda U.T S.T V p
				rhs2.setZero();
				for (auto& g : m_snapshotGroups)
					g.approximateRHS(m_positionsUsedVs, rhs2, m_collidedVerts);
				addAdditionalConstraints(m_positionsUsedVs, rhs2, m_collidedVerts);
			}
			else if (m_usePosSnapBases && !m_usingSkinSubspaces) {
				//std::cout << "compute LBS local step and project it to POD pos subspace" << std::endl;
				// here  no projection to Position subspace m_rhsInterpol = lambda S.T V p
				m_rhsInterpol.setZero(m_positions.rows(), 3);
				for (auto& g : m_snapshotGroups){
					g.approximateRHS(m_positionsUsedVs, m_rhsInterpol, m_collidedVerts);  //  m_rhsInterpol = lambda S.T V p (computed for the samples on;y)
				}
				addAdditionalConstraints(m_positionsUsedVs, m_rhsInterpol, m_collidedVerts);
			}
			else { // here  no projection to Position subspace m_rhsInterpol = lambda S.T V p    no pos reduction (computed for full positions)
				m_rhsInterpol.setZero(m_positions.rows(), 3);
				for (auto& g : m_snapshotGroups)
					g.approximateRHS(m_positions, m_rhsInterpol, m_collidedVerts);
				addAdditionalConstraints(m_positions, m_rhsInterpol, m_collidedVerts);
			}
			
			//std::cout << m_rhsInterpol.rows() << std::endl;//-------------------------------------> reurns zero!!
			
			m_localStepOnlyProjectStopWatch.stopStopWatch();
			m_localStepRestStopWatch.startStopWatch();
			
			//std::cout << "Local step approximatedvia fitting method projected to POD" << std::endl;
		}
		else if(m_usingQDEIMComponents){
		
			m_rhs.setZero();
			if(m_solveDeimLS){
				for (auto& g : m_snapshotGroups){
				
					g.qdeimApproximateRHS(m_positions, m_rhs, m_collidedVerts);   //m_rhs = lambda S.T V (P^T V)^{-1} P^T p 
				}
				addAdditionalConstraints(m_positions, m_rhs, m_collidedVerts);
			}
			else{	// use deim interpolationBlocks directly
				
				int p = 1;   // constraint size
				if (m_TetStrainOnly) {
					p = 3;
				}
				std::vector< PDPositions > qdeimAuxilariesX(m_numQDEIMModes), qdeimAuxilariesY(m_numQDEIMModes), qdeimAuxilariesZ(m_numQDEIMModes);
				//std::vector< PDScalar > curWeightx(m_numQDEIMModes), curWeighty(m_numQDEIMModes), curWeightz(m_numQDEIMModes);
				
				PDPositions reducedAuxilaryX(deimBasisCols, 3), reducedAuxilaryY(deimBasisCols, 3), reducedAuxilaryZ(deimBasisCols, 3);
				if(m_usePosSnapBases){
					
					// Compute projections
					m_localStepOnlyProjectStopWatch.startStopWatch();
		//#pragma omp parallel for num_threads(PROJ_DYN_NUM_THREADS)                             // TODO: first run sim properly then paralellize                 
					for (int ind = 0; ind < m_numQDEIMModes; ind++) {
						ProjDynConstraint* cx = usedConstraints->at(m_SqdeimX[ind]);					
						int didCollide = -1;
						qdeimAuxilariesX[ind] = cx->getP(m_positions, didCollide);
						if (didCollide >= 0) m_collidedVerts[didCollide] = true;
						//curWeightx[ind] = usedConstraints->at(m_SqdeimX[ind])->getWeight();

						reducedAuxilaryX.block(ind * p, 0, p, 3) = qdeimAuxilariesX[ind];  //*curWeightx[ind];
					}
					
					for (int ind = 0; ind < m_numQDEIMModes; ind++) {
						ProjDynConstraint* cy = usedConstraints->at(m_SqdeimY[ind]);					
						int didCollide = -1;
						qdeimAuxilariesY[ind] = cy->getP(m_positions, didCollide);
						if (didCollide >= 0) m_collidedVerts[didCollide] = true;
						//curWeighty[ind] = usedConstraints->at(m_SqdeimY[ind])->getWeight();

						reducedAuxilaryY.block(ind * p, 0, p, 3) = qdeimAuxilariesY[ind];    //*curWeighty[ind];
						//std::cout << reducedAuxilaryY.block(ind*p, 0, p, 3) << std::endl;	
					}
					
					for (int ind = 0; ind < m_numQDEIMModes; ind++) {
						ProjDynConstraint* cz = usedConstraints->at(m_SqdeimZ[ind]);					
						int didCollide = -1;
						qdeimAuxilariesZ[ind] = cz->getP(m_positions, didCollide);
						if (didCollide >= 0) m_collidedVerts[didCollide] = true;
						//curWeightz[ind] = usedConstraints->at(m_SqdeimZ[ind])->getWeight();

						reducedAuxilaryZ.block(ind * p, 0, p, 3) = qdeimAuxilariesZ[ind];  //* curWeightz[ind];
					}
					m_localStepOnlyProjectStopWatch.stopStopWatch();
					m_localStepRestStopWatch.startStopWatch();

					m_rhs.resize(m_numPosPODModes+1, 3);
					
					m_rhs.col(0) = UTSTMx * reducedAuxilaryX.col(0);
					m_rhs.col(1) = UTSTMy * reducedAuxilaryY.col(1);
					m_rhs.col(2) = UTSTMz * reducedAuxilaryZ.col(2); 				
				}
				else{
					std::cout << "Fatal Error! QDEIM can not yet be used without position subspace reduction" << std::endl;
					return;
				} 
			
			}
		
		}
		else{ // Full local step (find projection p): no rhsInterpolation
			
			// used in case storing nonlinear snapshots "p"
			

			// Compute projections
			m_localStepOnlyProjectStopWatch.startStopWatch();
	#pragma omp parallel for num_threads(PROJ_DYN_NUM_THREADS)
			for (int ind = 0; ind < numConstraints; ind++) {
				ProjDynConstraint* c = usedConstraints->at(ind);
				int didCollide = -1;
				currentAuxilaries[ind] = c->getP(m_positions, didCollide);
				if (didCollide >= 0) m_collidedVerts[didCollide] = true;
			}	
			m_localStepOnlyProjectStopWatch.stopStopWatch();
			m_localStepRestStopWatch.startStopWatch();
			
			if(recordingPSnapshots){
			unsigned int auxLength, auxSize;
			PDMatrixRM nonlinAuxilary;
			auxLength = currentAuxilaries[0].size() ;
			auxSize = currentAuxilaries[0].rows() ;
			PDPositions nonlinearSnapshots;
			nonlinAuxilary.resize(numConstraints, auxLength);
			
			int numRecordedFrames = 600;
			nonlinearSnapshots.resize(numConstraints* auxSize,  3);
				for (int ind = 0; ind < numConstraints; ind++){
				
					PDPositions Atemp; //col major
					Atemp = currentAuxilaries[ind];
					//if(m_frameCount == 0 && ind < 5) std::cout << Atemp << std::endl;
					Eigen::Map<Eigen::VectorXd> vFlat(Atemp.data(),Atemp.size());
					nonlinAuxilary.row(ind) = vFlat;  //flatened row, respects in the columns major of currentAuxilaries
					//std::cout << nonlinAuxilary.row(ind).size() << std::endl;
					if(m_frameCount < numRecordedFrames && i == numIterations -1){
						if(m_frameCount == 10 && ind < 3) std::cout << currentAuxilaries[ind] << std::endl;
						nonlinearSnapshots.block(ind*auxSize, 0, auxSize, 3) = currentAuxilaries[ind];
						//std::cout << nonlinearSnapshots.rows() << " " << nonlinearSnapshots.cols() << std::endl;
					}
				}
				/* Choose the way to store the constraints projections, eiter stacked blocks or flattened */
				if(m_frameCount == 10) {
					std::cout << nonlinearSnapshots << std::endl;
					//PD::storePosBinary(nonlinearSnapshots, "/home/shaimaa/libigl/tutorial/doubleHRPD/meshFrames_fullSimulationsOFF/bunny/tetConstraintSnapshots/axuxiliries_stacked/" +PD::getMeshFileName(m_meshURL, "_aux_"+ std::to_string(m_frameCount)+".bin"));
					//PD::storeBaseBinaryRM(nonlinAuxilary, "/home/shaimaa/libigl/tutorial/doubleHRPD/meshFrames_fullSimulationsOFF/bunny/tetConstraintSnapshots/onlyAuxiliries_flaten/" +PD::getMeshFileName(m_meshURL, "_aux_"+ std::to_string(m_frameCount)+".bin"));
				}
				
			}
			
			// Sum projections
			m_rhs.setZero();
			m_constraintSummationStopWatch.startStopWatch();

			
			int d = 0;
			PROJ_DYN_PARALLEL_FOR
				for (d = 0; d < 3; d++) {
					for (int mind = 0; mind < numConstraints; mind++) {
						PDScalar curWeight = usedConstraints->at(mind)->getWeight();
						fastDensePlusSparseTimesDenseCol(m_rhs, usedConstraints->at(mind)->getSelectionMatrixTransposed(), currentAuxilaries[mind], d, curWeight);
						// here m_rhs = lambda S.T p
					}
				}
			m_constraintSummationStopWatch.stopStopWatch();
		}

		m_momentumStopWatch.startStopWatch();
		
		// Add the term from the conservation of momentum 	
		if(m_usingSkinSubspaces && !m_usePosSnapBases){
	
			// If using subspaces, transform the r.h.s to the subspace
			// (which is already the case when using rhs interpolation)
			if (!m_rhsInterpolation) {
				if (m_useSparseMatricesForSubspace) {
					rhs2 = m_baseFunctionsTransposedSparse * m_rhs;  // here m_rhs = lambda S.T p ---> rhs2 = U.T lambda S.T p 
				}
				else {
					rhs2 = m_baseFunctionsTransposed * m_rhs;
				}
			} // now m_rhs = lambda U.T S.T p

			// Now add the term from the conservation of momentum, where s is already in the subspace,
			// and the product of the subspace matrix and M/h^2 has been evaluated already.
			PROJ_DYN_PARALLEL_FOR
				for (int d = 0; d < 3; d++) {
					if (m_useSparseMatricesForSubspace) {
						rhs2.col(d) += m_rhsFirstTermMatrixSparse * s.col(d);
					}
					else {
						rhs2.col(d) += m_rhsFirstTermMatrix * s.col(d);
					}
				}
			
		}
		else if(m_usePosSnapBases && !m_usingSkinSubspaces){
			// If using subspaces, transform the r.h.s to the subspace
			// i.e we compute rhs2 =  (U.T M U s/ h^2) + lambda U.T S.T p 
			rhs2.setZero(m_baseXFunctionsTransposed.rows(), 3);
			if (!m_rhsInterpolation && !m_usingQDEIMComponents) {
				if (m_useSparseMatricesForSubspace) {
					if(isPODBasisOrthogonal){
						#pragma omp parallel
						#pragma omp single nowait
						{
						#pragma omp task
							rhs2.col(0) = m_baseXFunctionsTransposedSparse * m_rhs.col(0);   // m_rhs = lambda S.T p here is computed through a full local step
						#pragma omp task                                                                           // rhs2 = lambda U.T S.T p 
							rhs2.col(1) = m_baseYFunctionsTransposedSparse * m_rhs.col(1);
						#pragma omp task
							rhs2.col(2) = m_baseZFunctionsTransposedSparse * m_rhs.col(2);
						} 
					}
					else{
						#pragma omp parallel
						#pragma omp single nowait
						{
						#pragma omp task
							rhs2.col(0) = m_baseXFunctionsTransposedSparse * m_rhs.col(0);   // m_rhs = lambda S.T p here is computed through a full local step
						#pragma omp task                                                         // rhs2 = lambda U.T S.T p 
							rhs2.col(1) = m_baseYFunctionsTransposedSparse * m_rhs.col(1);
						#pragma omp task
							rhs2.col(2) = m_baseZFunctionsTransposedSparse * m_rhs.col(2);
						} 
					}
				}
				else {
					// TODO: error!
					//projectToPODSubspace(rhs2, m_rhs, isPODBasisOrthogonal);
					// now rhs2 = lambda U.T S.T p 
				}
			
			}
			else if (m_rhsInterpolation && !m_usingQDEIMComponents) { //m_rehInterpolation with POD pos
				rhs2.setZero(m_baseXFunctionsTransposed.rows(), 3);
				#pragma omp parallel
					#pragma omp single nowait
					{
					#pragma omp task
						rhs2.col(0) = m_baseXFunctionsTransposedSparse * m_rhsInterpol.col(0);   // m_rhsInterpol = lambda S.T V p here is computed through a LBS local step
					#pragma omp task                                                                 // rhs2 = lambda U.T S.T V p 
						rhs2.col(1) = m_baseYFunctionsTransposedSparse * m_rhsInterpol.col(1);
					#pragma omp task
						rhs2.col(2) = m_baseZFunctionsTransposedSparse * m_rhsInterpol.col(2);
					}
			
			}
			else{ // m_usingQDEIMComponents in the LS solve with POD
				if(m_solveDeimLS){
				#pragma omp parallel
						#pragma omp single nowait
						{
						#pragma omp task
							rhs2.col(0) = m_baseXFunctionsTransposedSparse * m_rhs.col(0);   // m_rhs = lambda S.T V (P^T V)^{-1} P^T p 
						#pragma omp task                                                         // rhs2 = lambda U.T S.T V (P^T V)^{-1} P^T p 
							rhs2.col(1) = m_baseYFunctionsTransposedSparse * m_rhs.col(1);
						#pragma omp task
							rhs2.col(2) = m_baseZFunctionsTransposedSparse * m_rhs.col(2);
						} 
				}
				else{  // m_usingQDEIMComponents (no LS solve) with POD
					#pragma omp parallel
						#pragma omp single nowait
						{
						#pragma omp task
							rhs2.col(0) = m_rhs.col(0);   // m_rhs = lambda U.T S.T V (P^T V)^{-1} P^T p 
						#pragma omp task                      // rhs2 = m_rhs
							rhs2.col(1) = m_rhs.col(1);
						#pragma omp task
							rhs2.col(2) = m_rhs.col(2);
						} 
				}	
			
			}
			
			// Now add the term from the conservation of momentum, where s is already in the subspace,
			// and the product of the subspace matrix and M/h^2 has been evaluated already.
			if (m_useSparseMatricesForSubspace) {					
					#pragma omp parallel
					#pragma omp single nowait
					{
					#pragma omp task
						rhs2.col(0) += m_rhsXFirstTermMatrixSparse * s.col(0);  //  // rhs2 = lambda U.T S.T M p + U.T (M/h^2)  s
					#pragma omp task
						rhs2.col(1) += m_rhsYFirstTermMatrixSparse * s.col(1);
					#pragma omp task
						rhs2.col(2) += m_rhsZFirstTermMatrixSparse * s.col(2);
					} 
					
				}
				else {
					rhs2.col(0) += m_rhsXFirstTermMatrix * s.col(0);
					rhs2.col(1) += m_rhsYFirstTermMatrix * s.col(1);
					rhs2.col(2) += m_rhsZFirstTermMatrix * s.col(2);
				}
				
				
			
			// now rhs2 = lambda U.T S.T p + (U.T M U s/ h^2)	
			// and we have from setup()
			// m_lhsMatrixSampled = (1/ h^2) U.T M U + Sum_i U.T lambda_i S_i.T S_i U
			// In the global solve we have
			// m_lhsMatrixSampled * m_positionsSubspace = rhs2   (for m_positionsSubspace)
		}	
		
		else if(!m_usingPosSubspaces && !recordingSTpSnapshots){  
			// if not reducing position space
			// If no subspaces are used, the full rhs from the constraints just needs to be added
			// to the conservation of momentum terms so that m_rhs = (M/h^2 ) s + lambda S.T p
			// while we are are recording constraints snapshots then stays m_rhs = lambda S.T p
			int v = 0;
			if(!m_rhsInterpolation){
			PROJ_DYN_PARALLEL_FOR
				for (v = 0; v < m_numVertices; v++) {
					for (int d = 0; d < 3; d++) {
						m_rhs(v, d) += m_rhsMasses(v) * s(v, d);   // here m_rhs = lambda S.T p + (M/h^2) s
 					}
				}
			}
			else{
			PROJ_DYN_PARALLEL_FOR
				for (v = 0; v < m_numVertices; v++) {
					for (int d = 0; d < 3; d++) {
						m_rhsInterpol(v, d) += m_rhsMasses(v) * s(v, d);   // here m_rhsInterpol = lambda S.T V p + (M/h^2) s
 					}
				}
			}
		}


		m_momentumStopWatch.stopStopWatch();
		m_localStepRestStopWatch.stopStopWatch();
		m_localStepStopWatch.stopStopWatch();
		//std::cout << "Full local step done, now we go to global step................." << std::endl;
		//************************************
		// Global step: Solve the linear system with fixed constraint projections
		//************************************
		m_globalStepStopWatch.startStopWatch();
		// Solve, for x, y and z in parallel	
		if (m_usingPosSubspaces){
			if(m_usingSkinSubspaces && !m_usePosSnapBases){
				// Only subsspace positions are updated, the full positions are only evaluated
				// where the constraints need them.
				int d = 0;
				PROJ_DYN_PARALLEL_FOR
				for (d = 0; d < 3; d++) {
					if (m_useSparseMatricesForSubspace) {
						m_positionsSubspace.col(d) = m_subspaceSystemSolverSparse.solve(rhs2.col(d));
						
					}
					else {
						m_positionsSubspace.col(d) = m_denseSolver.solve(rhs2.col(d));
					}
				}
			}
			else if(m_usePosSnapBases && !m_usingSkinSubspaces){
				// we solve for the subPostions: (U.T M U/h^2  + lambda U.T S.T S U) q = lambda U.T S.T p + (U.T M U s/ h^2)
				if (m_useSparseMatricesForSubspace) {
					//std::cout << "Solving global system using sprse solvers" << std::endl;
					
					#pragma omp parallel
					#pragma omp single nowait
					{
					#pragma omp task
						m_positionsSubspace.col(0) = m_subspaceXSystemSolverSparse.solve(rhs2.col(0));
					#pragma omp task
						m_positionsSubspace.col(1) = m_subspaceYSystemSolverSparse.solve(rhs2.col(1));
					#pragma omp task
						m_positionsSubspace.col(2) = m_subspaceZSystemSolverSparse.solve(rhs2.col(2));
					}
					
					
					if(m_subspaceXSystemSolverSparse.info()!= Eigen::Success || m_subspaceYSystemSolverSparse.info()!= Eigen::Success  || m_subspaceZSystemSolverSparse.info()!= Eigen::Success  ) {
					  // solving failed
					  std::cout << "FATAL ERROR! sparse slovers for the global step failed!" << std::endl;
					  return;
					}
				}
				else {
					//std::cout << "Solving global system using dense solvers" << std::endl;
					m_positionsSubspace.col(0) = m_denseXSolver.solve(rhs2.col(0));
					m_positionsSubspace.col(1) = m_denseYSolver.solve(rhs2.col(1));
					m_positionsSubspace.col(2) = m_denseZSolver.solve(rhs2.col(2));
					
					/*
					#pragma omp parallel
					#pragma omp single nowait
					{
					#pragma omp task
						m_positionsSubspace.col(0) = m_denseXSolver.solve(rhs2.col(0));
					#pragma omp task
						m_positionsSubspace.col(1) = m_denseYSolver.solve(rhs2.col(1));
					#pragma omp task
						m_positionsSubspace.col(2) = m_denseZSolver.solve(rhs2.col(2));
					}*/
					if(m_denseXSolver.info()!= Eigen::Success || m_denseYSolver.info()!= Eigen::Success  || m_denseZSolver.info()!= Eigen::Success  ) {
					  // solving failed
					  std::cout << "FATAL ERROR! dense slovers for the global step failed!" << std::endl;
					  return;
					}
				}
			}
			else{
				std::cout << " Error: Global step not yet possible with this position reduction method!" << std::endl;
				return;
			}
			
			
		
		}
		if (!m_usingPosSubspaces) {
			int d = 0;
		
			if(!m_rhsInterpolation){
			PROJ_DYN_PARALLEL_FOR
				for (d = 0; d < 3; d++) {
					m_positions.col(d) = m_linearSolver.solve(m_rhs.col(d));
					// This solves: for full original system (M/h^2 + Sum S.T S) q =  (M/h^2) s + lambda S.T p
					//		and for nonlinear part only (M/h^2 + Sum S.T S) q =  lambda S.T p  (recording QDEIM snapshots)
				}
			}
			else{
			PROJ_DYN_PARALLEL_FOR
				for (d = 0; d < 3; d++) {
					m_positions.col(d) = m_linearFullLHSinterploRHSSolver.solve(m_rhsInterpol.col(d));
					// This solves: for full system with rhsInterpolation only(M/h^2 + Sum S.T S) q =  (M/h^2) s + lambda S.T V p
					//		and for nonlinear part only (M/h^2 + Sum S.T S) q =  lambda S.T p  (recording QDEIM snapshots)
				}
			}
		}
		/*else if(recordingSTpSnapshots){
		int d = 0;
			PROJ_DYN_PARALLEL_FOR
			for (d = 0; d < 3; d++) {
				m_positions.col(d) = m_rhs.col(d); //  here we basically have: (M/h^2 ) q =  lambda S.T p  
			}
		}*/
			
		m_globalStepStopWatch.stopStopWatch();

		// Partially update vertex positions (vertices involved in the evaluation of constraints)
		// Does not need to be done in the last iteration, since vertex positions are fully
		// updated after.
		m_updatingVPosStopWatch.startStopWatch();
		if (m_usingPosSubspaces && !(i == numIterations - 1)) {
			if (m_usingSkinSubspaces && m_rhsInterpolation && !m_usePosSnapBases) {
				updatePositionsSampling(m_positionsUsedVs, m_positionsSubspace, true);  // update for used vertcies only
			}
			else if(m_usePosSnapBases){
				if (!m_rhsInterpolation) updatePODPositionsSampling(m_positions, m_positionsSubspace, podUsedVerticesOnly);       // no local supprt
				else if (m_rhsInterpolation) updatePODPositionsSampling(m_positionsUsedVs, m_positionsSubspace, podUsedVerticesOnly);
			}
			else {
				updatePositionsSampling(m_positions, m_positionsSubspace, false);       // update for all vertices
			}
		}  /// how this update is done for full simulations?
		m_updatingVPosStopWatch.stopStopWatch();

	} // End of local global loop

	
	if (m_rhsInterpolation && m_collisionFreeDraw) {
		if(m_usingSkinSubspaces){
			updatePositionsSampling(m_positionsUsedVs, m_positionsSubspace, true);
			handleGripAndCollisionsUsedVs(s, false);
		}
		
	}
	else if (m_rhsInterpolation) {
		if(m_usingSkinSubspaces) s = m_positionsSubspace;
		if(m_usePosSnapBases){
			 s = m_positionsSubspace;
			 updatePODPositionsSampling(m_positionsUsedVs, m_positionsSubspace, podUsedVerticesOnly);
		}
		handleGripAndCollisionsUsedVs(s, false);
	}
	else if (!m_rhsInterpolation && m_usingSkinSubspaces){
		updatePositionsSampling(m_positionsUsedVs, m_positionsSubspace, true);
		handleGripAndCollisionsUsedVs(s, false);
	}
	
	m_surroundingBlockStopWatch.startStopWatch();

	// Evaluate full positions once so that outside methods
	// have access to them (for e.g. displaying the mesh)
	m_fullUpdateStopWatch.startStopWatch();
	
	
	if (m_usingSkinSubspaces && !m_usePosSnapBases) {
		if (m_useSparseMatricesForSubspace) {
#ifdef PROJ_DYN_USE_CUBLAS
			if (m_vPosGPUUpdate) {
				// The vertex position update is done on the GPU
				PDScalar one = 1.;
				for (int d = 0; d < 3; d++) {
					m_vPosGPUUpdate->mult(s.data() + (d * s.rows()), nullptr, one, false, d, m_numOuterVertices);
				}
			}
			else
#endif
			if (!m_parallelVUpdate) {
			
				// The vertex update is still done in parallel, but only for the three columns
				PROJ_DYN_PARALLEL_FOR
					for (int d = 0; d < 3; d++) {
						m_positions.col(d) = m_baseFunctionsSparse * s.col(d);
					}
			}
			else {
			
				// The vertex update is done in finer granularity, by splitting the vertex
				// positions into blocks of n rows and building the vector from this
				int blockSize = m_baseFunctionsSparseBlocks[0].rows();
				int numBlocks = std::ceil((float)m_positions.rows() / (float)blockSize);
				PROJ_DYN_PARALLEL_FOR
					for (int b = 0; b < numBlocks; b++) {
						int curSize = blockSize;
						if (b == numBlocks - 1) {
							curSize = blockSize - (numBlocks * blockSize - m_positions.rows());
						}
						m_positions.block(b*blockSize, 0, curSize, 3) = m_baseFunctionsSparseBlocks[b] * s;
					}
			}
		}
		else {
			PROJ_DYN_PARALLEL_FOR
				for (int d = 0; d < 3; d++) {
					m_positions.col(d) = m_baseFunctions * s.col(d);
				}
		}
	}
	
	if (m_usePosSnapBases && !m_usingSkinSubspaces) {
		if (m_useSparseMatricesForSubspace) {
#ifdef PROJ_DYN_USE_CUBLAS
			if (m_vPosGPUUpdate) {
				// The vertex position update is done on the GPU
				std::cout << "NOT yet availablr for POD spaces" << std::endl;
				return;
			}
			else
#endif
			if (!m_parallelVUpdate) {
				// The vertex update is still done in parallel, but only for the three columns
								
				#pragma omp parallel
				#pragma omp single nowait
				{
				#pragma omp task
					m_positions.col(0) = m_baseXFunctionsSparse * s.col(0);
				#pragma omp task
					m_positions.col(1) = m_baseYFunctionsSparse * s.col(1);
				#pragma omp task
					m_positions.col(2) = m_baseZFunctionsSparse * s.col(2);
				}
				
			}
			else {		
				// The vertex update is done in finer granularity, by splitting the vertex
				// positions into blocks of n rows and building the vector from this
				int blockSize = m_baseFunctionsSparseBlocks[0].rows();
				int numBlocks = std::ceil((float)m_positions.rows() / (float)blockSize);
				PROJ_DYN_PARALLEL_FOR
					for (int b = 0; b < numBlocks; b++) {
						int curSize = blockSize;
						if (b == numBlocks - 1) {
							curSize = blockSize - (numBlocks * blockSize - m_positions.rows());
						}						
						#pragma omp parallel
						#pragma omp single nowait
						{
						#pragma omp task
							m_positions.col(0).block(b*blockSize, 0, curSize, 3) = m_baseXFunctionsSparseBlocks[b] * s.col(0);
						#pragma omp task
							m_positions.col(1).block(b*blockSize, 0, curSize, 3) = m_baseYFunctionsSparseBlocks[b] * s.col(1);
						#pragma omp task
							m_positions.col(2).block(b*blockSize, 0, curSize, 3) = m_baseZFunctionsSparseBlocks[b] * s.col(2);
						}
					}
			}
		}
		else { // case not m_useSparseMatricesForSubspace
		
			updatePODPositionsSampling(m_positions, s, podUsedVerticesOnly);
		}
	}
	
	m_fullUpdateStopWatch.stopStopWatch();

	// Update Velocities
	if (m_usingPosSubspaces) {
	
		if (!m_rhsInterpolation) {
			// Update full velocities
			m_velocities = (m_positions - oldFullPos) / m_timeStep;
		}
		// Update subspace velocities as well
		m_velocitiesSubspace = (m_positionsSubspace - oldPos) / m_timeStep;
	}
	else {
		m_velocities = (m_positions - oldPos) / m_timeStep;
	}

	m_frameCount++;
	std::cout << "Frame: " << m_frameCount << std::endl;

	std::vector<unsigned int>* usedVerts = &m_allVerts;    // usedVerts is a pointer to a vector stores vertices indicies
	if (m_rhsInterpolation) {
		usedVerts = &m_constraintVertexSamples;        // OR usedVerts is a pointer to a vector stores constrianed vertices indicies
	}

	m_surroundingBlockStopWatch.stopStopWatch();
	m_totalStopWatch.stopStopWatch();
	
	/*
	// This part so that we can see the object at center of the screen
	PDPositions tempPos = m_positions;
	for (int d = 0; d < 3; d++) {
		PDScalar avg = tempPos.col(d).sum() / tempPos.rows();
		for (int v = 0; v < m_numVertices; v++)
			tempPos(v, d) -= avg;
	}
	m_positions = tempPos/(0.01*tempPos.norm());	 
	*/
	if (STORE_FRAMES_OFF)
	{
		igl::writeOFF(m_meshSnapshotsDirectory + "pos_" + std::to_string(m_frameCount) + ".off", m_positions, m_triangles);
	}
}

/*
PD::CollisionObject::CollisionObject()
	:
	m_type(-1),
	m_s1(0),
	m_s2(0),
	m_s3(0),
	m_pos(0, 0, 0)
{
	std::cout << "===PD::CollisionObject::CollisionObject nothing===" << std::endl;
}

PD::CollisionObject::CollisionObject(int type, PD3dVector& pos, PDScalar s1, PDScalar s2, PDScalar s3)
	:
	m_type(type),
	m_s1(s1),
	m_s2(s2),
	m_s3(s3),
	m_pos(pos)
{
	std::cout << "===PD::CollisionObject::CollisionObject===" << std::endl;
}
*/
bool PD::CollisionObject::resolveCollision(PD3dVector& newPos)
{
	
	PD3dVector relPos;
	switch (m_type) {
	case CollisionType::sphere:
		if ((newPos - m_pos).norm() < m_s1) {
			PD3dVector dir = (newPos - m_pos);
			dir.normalize();
			dir *= m_s1;
			newPos = m_pos + dir;
			return true;
		}
		break;
	case CollisionType::floor:
		if (newPos(1) < m_s1) {
			newPos(1) = m_s1;
			return true;
		}
		break;
	case CollisionType::block:
		relPos = newPos - m_pos;
		if (relPos(0) > 0 && relPos(0) < m_s1) {
			if (relPos(1) > 0 && relPos(1) < m_s2) {
				if (relPos(2) > 0 && relPos(2) < m_s3) {

					PDScalar corX = 0, corY = 0, corZ = 0;
					if (relPos(0) < m_s1 - relPos(0)) {
						corX = -relPos(0);
					}
					else {
						corX = m_s1 - relPos(0);
					}

					if (relPos(1) < m_s2 - relPos(1)) {
						corY = -relPos(1);
					}
					else {
						corY = m_s2 - relPos(1);
					}

					if (relPos(2) < m_s3 - relPos(2)) {
						corZ = -relPos(2);
					}
					else {
						corZ = m_s3 - relPos(2);
					}

					if (std::abs(corX) <= std::abs(corY) && std::abs(corX) <= std::abs(corZ)) {
						newPos(0) += corX;
					}
					else if (std::abs(corY) <= std::abs(corX) && std::abs(corY) <= std::abs(corZ)) {
						newPos(1) += corY;
					}
					else if (std::abs(corZ) < std::abs(corX) && std::abs(corZ) < std::abs(corY)) {
						newPos(2) += corZ;
					}

					return true;
				}
			}
		}
		break;
	default:
		return false;
	}
	return false;
}
/*
void PD::ProjDynSimulator::addCollisionsFromFile(std::string fileName)
{
	std::cout << "===PD::CollisionObject::addCollisionsFromFile===" << std::endl;
	// Read collisions file
	std::ifstream nodeFile;
	nodeFile.open(fileName);
	if (nodeFile.is_open()) {
		std::cout << "Reading and adding collision objects from the " << fileName << "..." << std::endl;
		std::string line;
		while (std::getline(nodeFile, line))
		{
			std::istringstream iss(line);
			char typeC;
			PDScalar x = 0, y = 0, z = 0;
			PDScalar s1 = 0, s2 = 0, s3 = 0;
			iss >> typeC >> x >> y >> z >> s1 >> s2 >> s3;
			int type = -1;
			switch (typeC) {
			case 's':
				type = CollisionType::sphere;
				break;
			case 'f':
				type = CollisionType::floor;
				break;
			case 'b':
				type = CollisionType::block;
				break;
			}
			PD3dVector pos = PD3dVector(x, y, z);
			CollisionObject col(type, pos, s1, s2, s3);
			m_collisionObjects.push_back(col);
		}
	}
}

void PD::ProjDynSimulator::setEnforceCollisionFreeDraw(bool enable)
{
	std::cout << "===PD::CollisionObject::setEnforceCollisionFreeDraw===" << std::endl;
	m_collisionFreeDraw = enable;
}
*/
void PD::ProjDynSimulator::setStiffnessFactor(PDScalar w)
{
	//std::cout << "===PD::CollisionObject::setStiffnessFactor===" << std::endl;
	for (auto& g : m_snapshotGroups) {
		g.setWeightFactor(w);
	}

	m_stiffnessFactor = w;
	refreshLHS();
}

void PD::ProjDynSimulator::handleGripAndCollisionsUsedVs(PDPositions& s, bool update)
{
	// Detect and resolve collisions on evaluated vertices
	m_collisionCorrection = false;
	//m_planeBounceCorrection = false;
	if(m_usingSkinSubspaces){
		
		if(m_rhsInterpolation){
			PROJ_DYN_PARALLEL_FOR
				for (int v = 0; v < m_positionsUsedVs.rows(); v++) {
					resolveCollision(v, m_positionsUsedVs, m_positionCorrectionsUsedVs);
				}

			
			// Change position of gripped vertices
			if (update) {
				if (m_grippedVertices.size() > 0) {
					for (unsigned int i = 0; i < m_grippedVertices.size(); i++) {
						if (m_usedVertexMap[m_grippedVertices[i]] >= 0) {
							m_positionsUsedVs.row(m_usedVertexMap[m_grippedVertices[i]]) = m_grippedVertexPos.row(i);
						}
					}
				}
			}

			// If collisions were resolved, update s in the subspace via interpolation of the corrected vertices
			if (m_collisionCorrection || m_grippedVertices.size() > 0) {
				PROJ_DYN_PARALLEL_FOR
					for (int d = 0; d < 3; d++) {
						if (m_useSparseMatricesForSubspace) {
							s.col(d) = m_usedVertexInterpolatorSparse.solve(m_usedVertexInterpolatorRHSMatrixSparse * m_positionsUsedVs.col(d));
						}
						else {
							s.col(d) = m_usedVertexInterpolator.solve(m_usedVertexInterpolatorRHSMatrix * m_positionsUsedVs.col(d));
						}
					}
				if (update) m_positionsSubspace = s;
				updatePositionsSampling(m_positionsUsedVs, m_positionsSubspace, true);
			}
		}
		else{ //no m_rhsInterpolation
			PROJ_DYN_PARALLEL_FOR
				for (int v = 0; v < m_positions.rows(); v++) {
					resolveCollision(v, m_positions, m_positionCorrections);
				}
				
			// Change position of gripped vertices
			if (update) {
				if (m_grippedVertices.size() > 0) {
					for (unsigned int i = 0; i < m_grippedVertices.size(); i++) {
						if (m_usedVertexMap[m_grippedVertices[i]] >= 0) {
							m_positionsUsedVs.row(m_usedVertexMap[m_grippedVertices[i]]) = m_grippedVertexPos.row(i);
						}
					}
				}
			}
			// If collisions were resolved, update s in the subspace via interpolation of the corrected vertices
			if (m_collisionCorrection || m_grippedVertices.size() > 0) {
				PROJ_DYN_PARALLEL_FOR
					for (int d = 0; d < 3; d++) {
						if (m_useSparseMatricesForSubspace) {
							s.col(d) = m_usedVertexInterpolatorSparse.solve(m_usedVertexInterpolatorRHSMatrixSparse * m_positionsUsedVs.col(d));
						}
						else {
							s.col(d) = m_usedVertexInterpolator.solve(m_usedVertexInterpolatorRHSMatrix * m_positionsUsedVs.col(d));
						}
					}
				if (update) m_positionsSubspace = s;
				
				updatePositionsSampling(m_positions, m_positionsSubspace, false);
			} 
				
			projectToSubspace(m_positionsSubspace, m_positions, false);   
			//projectToSubspace(m_velocitiesSubspace, m_velocities, false);
		
		}
		
	}
	else if(m_usePosSnapBases){
		if(m_rhsInterpolation){
			PROJ_DYN_PARALLEL_FOR
				for (int v = 0; v < m_positionsUsedVs.rows(); v++) {
					resolveCollision(v, m_positionsUsedVs, m_positionCorrectionsUsedVs);
				}
			// Change position of gripped vertices
			if (update) {
				if (m_grippedVertices.size() > 0) {
					for (unsigned int i = 0; i < m_grippedVertices.size(); i++) {
						if (m_usedVertexMap[m_grippedVertices[i]] >= 0) {
							m_positionsUsedVs.row(m_usedVertexMap[m_grippedVertices[i]]) = m_grippedVertexPos.row(i);
						}
					}
				}
			}
			// If collisions were resolved, update s in the subspace via interpolation of the corrected vertices
			if (m_collisionCorrection || m_grippedVertices.size() > 0) {
				
					
					if (m_useSparseMatricesForSubspace) {
						#pragma omp parallel
						#pragma omp single nowait
						{
						#pragma omp task
							s.col(0) = m_usedVertexXInterpolatorSparse.solve(m_usedVertexXInterpolatorRHSMatrixSparse * m_positionsUsedVs.col(0));
						#pragma omp task
							s.col(1) = m_usedVertexYInterpolatorSparse.solve(m_usedVertexYInterpolatorRHSMatrixSparse * m_positionsUsedVs.col(1));
						#pragma omp task
							s.col(2) = m_usedVertexZInterpolatorSparse.solve(m_usedVertexZInterpolatorRHSMatrixSparse * m_positionsUsedVs.col(2));
						}			
					}
					else {
						
						s.col(0) = m_usedVertexXInterpolator.solve(m_usedVertexXInterpolatorRHSMatrix * m_positionsUsedVs.col(0));
						s.col(1) = m_usedVertexYInterpolator.solve(m_usedVertexYInterpolatorRHSMatrix * m_positionsUsedVs.col(1));
						s.col(2) = m_usedVertexZInterpolator.solve(m_usedVertexZInterpolatorRHSMatrix * m_positionsUsedVs.col(2));
					}
					
				if (update) m_positionsSubspace = s;
				
				updatePODPositionsSampling(m_positions, m_positionsSubspace, podUsedVerticesOnly);
			}		
		
		}
		else{
			PROJ_DYN_PARALLEL_FOR
			for (int v = 0; v < m_positions.rows(); v++) {
				resolveCollision(v, m_positions, m_positionCorrections);
			}
			updatePODPositionsSampling(m_positionsUsedVs, m_positionsSubspace, false);
		}
	
	
	}
	
	
	
	
}

/* If the values m_timeStep or m_stiffnessFactor have changed, this method recomputes the matrices involved and
and updates their factorizations. */

void PD::ProjDynSimulator::refreshLHS()
{

	if (m_rhsInterpolation) {
		m_lhsMatrixSampled = m_subspaceLHS_mom * (1. / (m_timeStep * m_timeStep)) + m_stiffnessFactor * m_subspaceLHS_inner;
		if (!m_useSparseMatricesForSubspace) {
			m_denseSolver.compute(m_lhsMatrixSampled);
		}
		else {
			PDSparseMatrix lhsMatrixSampledSparse = m_lhsMatrixSampled.sparseView(0, PROJ_DYN_SPARSITY_CUTOFF_HIGH_PREC);
			m_subspaceSystemSolverSparse.compute(lhsMatrixSampledSparse);
		}
	}
}

