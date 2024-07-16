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
#pragma once
//#define EIGEN_USE_MKL_ALL true
// Using Cholmod (by the define below) requires suitesparse.
// Read the _README.txt for details.
//#define PROJ_DYN_USE_CHOLMOD

// Use CUDA/CUBLAS for precomputation and/or for vertex position
// updates. Read the _README.txt for details.
//#define PROJ_DYN_USE_CUBLAS_IN_PRE
//#define PROJ_DYN_USE_CUBLAS

//#include "Eigen/Core"

#ifdef PROJ_DYN_USE_CUBLAS
#include "CUDAMatrixVectorMult.h"
#endif

#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/SparseCholesky"
#include "Eigen/SparseLU"
#ifdef PROJ_DYN_USE_CHOLMOD
#include "Eigen/CholmodSupport"
#endif

#include "StopWatch.h"

#include "ProjDynTypeDef.h"
#include "ProjDynMeshSampler.h"
#include "ProjDynRHSInterpol.h"

#define PROJ_DYN_NUM_THREADS 6
#define PROJ_DYN_EIGEN_NUM_THREADS 6
#define PROJ_DYN_PARALLEL_FOR _Pragma("omp parallel for num_threads(PROJ_DYN_NUM_THREADS)") 

// If set to true, we sparsify the matrices used in the global step and for interpolation in the subspace
#define PROJ_DYN_SPARSIFY true
// Corrects mass of very small triangles to have a minimal mass
#define PROJ_DYN_MIN_MASS 1e-10f
// Entries below this value will be removed from matrices when sparsifying them
#define PROJ_DYN_SPARSITY_CUTOFF 1e-12
// Entries below this value will be removed from matrices when sparsifying high precision matrices
#define PROJ_DYN_SPARSITY_CUTOFF_HIGH_PREC 1e-20
// If using parallel blocks for updating vertex positions, this defines the size of those
#define PROJ_DYN_VPOS_BLOCK_SIZE 1000

// Setting results storage macros
// If storing frames in .png format (e.g. for numerical stability comparision)
#define STORE_FRAMES_PNG false
// If storing frames in .off format (e.g. for FOM snapshots collection, or error computation between different reduction methods)
#define STORE_FRAMES_OFF false

// characteristics, if we are using PCA bases for position space
#define SNAPBASES_POSITION_WEIGHTING "_Volkwein_Standarized"
#define SNAPBASES_POSITION_SUPPORT "_Local"
#define SNAPBASES_POSITION_ALIGNMENT "_centered"
#define SNAPBASES_POSITION_ORTHOGONAL "_nonOrthogonalized"

#define SNAPBASES_POSITION_SPARSE "_nonSparse"

namespace PD {

#ifndef PROJ_DYN_USE_CHOLMOD
	typedef Eigen::SimplicialLDLT<PD::PDSparseMatrix> PDSparseSolver;
#else
	typedef Eigen::CholmodSimplicialLDLT<PD::PDSparseMatrix> PDSparseSolver;
#endif
	
	struct Edge {
		unsigned int v1, v2;
		int t1, t2, vOtherT1, vOtherT2;
	};

	class ProjDynConstraint;
	class CenterConstraint;
	class TetExampleBased;

	class PDSparseVector;

	enum CollisionType {
		sphere,
		block,
		floor
	};

	class CollisionObject {
	private:
		int m_type;
		PD3dVector m_pos;
		PDScalar m_s1, m_s2, m_s3;
	public:
		CollisionObject();
		CollisionObject(int type, PD3dVector& pos, PDScalar s1, PDScalar s2, PDScalar s3);
		bool resolveCollision(PD3dVector& desiredPos);
	};

	class ProjDynSimulator {

	public:
		ProjDynSimulator(PDTriangles& triangles, 
			PDPositions& initialPositions,
			PDPositions& initialVelocities, PDScalar timeStep,
			int numPCAComponents = -1,
			std::string pca_directory = "",
			int numSPLOCSComponents = -1,
			std::string splocs_directory = "",
			int numLBSPosSamples = -1, 
			PDScalar baseFunctionRadius = 2.5,
			int interpolBaseSize = 120,
			PDScalar rhsInterpolWeightRadius = 2.,
			int numConstraintSamples = -1,
			PDScalar massPerUnitArea = 1, 
			PDScalar dampingAlpha = 0,
			bool makeTets = false,
			std::string meshURL = "",
			PDScalar rhsRegularizationWeight = 0.,
			PDScalar yTranslation = 0);

		void step(int numIterations);
		PDPositions& getPositions();
		PDPositions& getVelocities();
		int getNumVertices();
		void addConstraint(ProjDynConstraint* c, bool alwaysAdd = false);
		void setup();

		void opt_setup();
		void full_lhsSetup();
		void lbs_lhsSetup();
		void snapBases_lhsSetup();

		void setExternalForces(PDPositions fExt);
		void addGravity(PDScalar g);
		void addFloor(int floorCoordinate, PDScalar floorHeight, PDScalar floorCollisionWeight);
		void addEdgeSprings(PDScalar weight, PDScalar rangeMin, PDScalar rangeMax);
		void addTriangleStrain(PDScalar weight, PDScalar rangeMin, PDScalar rangeMax);
		void addTetStrain(PDScalar weight, PDScalar rangeMin, PDScalar rangeMax);
		void addBendingConstraints(PDScalar weight, bool preventBendingFlips, bool flatBending);
		void setFrictionCoefficient(PDScalar coeff, PDScalar rCoeff = -1.);
		void printTimeMeasurements();
		void addHandleConstraint(CenterConstraint* cc);
		void changeTimeStep(PDScalar newTimeStep);
		void resetPositions();
		void resetVelocities();

		void setGrip(std::vector<unsigned int>& vInds, PDPositions gripPos);
		std::vector<unsigned int>& getUsedVertices();
		PDPositions& getUsedVertexPositions();
		void releaseGrip();

		void addConstraintSample(ProjDynConstraint* c);
		void setExamplePoses(std::vector<PDPositions> exPoses, PDScalar weight, bool forSprings = false);
		PDPositions extendSurfaceDeformationToTets(PDPositions& surfacePos);
		void setExampleWeights(std::vector<PDScalar>& exWeights);
		~ProjDynSimulator();

		void setParallelVUpdateBlockSize(int);
		void setEnableVUpdate(bool);

		bool m_useSparseMatricesForSubspace;
		void initializeGPUVPosMapping(GLuint bufferId);


		void addCollisionsFromFile(std::string fileName);
		void setEnforceCollisionFreeDraw(bool enable);

		void setStiffnessFactor(PDScalar w);

		PDScalar evaluateEnergy(PDPositions& q, PDPositions& s);

		void setInitialPos(PDPositions& startPos);
		void setBlowup(PDScalar enable);

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		int m_frameCount;
		
		void addRandomFloor(PDScalar a, PDScalar b, PDScalar c, PDScalar d, PDScalar floorCollisionWeight);

		std::string m_meshName = "";
		std::string m_meshSnapshotsDirectory = "";

	private:
		PDScalar m_initPosNorm;

		std::string m_meshURL;

		PDScalar m_blowupStrength = 0;
		PDPositions m_vertexNormals;

		bool m_collisionFreeDraw = false;
		/*  Only one vertex per sampled constraint is selected.
			Used for self-collision detection. */
		std::vector< unsigned int > m_usedVerticesSlim;
		std::vector< unsigned int > m_slimToUsedIndices;
		PDTets m_simpleTets;

		std::vector< CollisionObject > m_collisionObjects;
		void handleGripAndCollisionsUsedVs(PDPositions& s, bool update);

		void refreshLHS();

		/* Positions and velocities, describing the state of the system */
		PDPositions m_positions;
		PDPositions m_velocities;
		/* Mesh information */
		int m_numTriangles, m_numVertices, m_numOuterVertices, m_numTets;
		/* Indices of corners of the triangles that make up the surface connectivity */
		PDTriangles m_triangles;
		/* Edge list, only available if addEdgeSpring was called */
		std::vector< std::pair< unsigned int, unsigned int > > m_edges;
		/* If available, indices of the corners of the tetrahedrons that make up
		the volumetric connectivity */
		PDTets m_tetrahedrons;
		bool m_hasTetrahedrons;
		/* Normalization multiplier for areas and volumes associated to vertices
		and elements such that they are 1 on average. */
		PDScalar m_normalization;

		ProjDynMeshSampler m_sampler;

		void makeEdges();

		bool m_flatBending;

		/* Example poses stuff */
		std::vector<PDPositions> m_exPoses;
		std::vector<ProjDynConstraint*> m_tetExConstraints;

		/* Surface deformation extension stuff */
		bool m_surfDefExtInit = false;
		PDSparseMatrix m_surfDefExtRHSMat;
		PDSparseMatrix m_surfDefExtFixedPartMat;
		PDSparseSolver m_surfDefExtSolver;
		std::vector<ProjDynConstraint*> m_surfDefExtConstraints;

		int m_parallelVUpdateBSize;
		bool m_parallelVUpdate;
		void updateParallelVUpdateBlocks();

#ifdef PROJ_DYN_USE_CUBLAS
		CUDAMatrixVectorMultiplier* m_vPosGPUUpdate = nullptr;
#endif

		int m_floorCoordinate;
		PDScalar m_floorCollisionWeight;
		PDScalar m_floorHeight;
		std::vector< std::vector< Edge > > m_vertexStars;
		std::vector< std::vector< unsigned int > > m_tetsPerVertex;
		/* Additional external per-vertex forces like wind, interavtive "grabbing", ..., can be set
		with setExternalForces(fExt) */
		PDPositions m_fExt;
		/* Gravitational per-vertex forces, can be added with addGravity(massPerUnitArea) */
		PDPositions m_fGravity;
		/* Contains external forces, gravitational forces and friction forces, all weighted by inverse
		vertex masses and multiplied by m_timeStep squared, such that they can be directly added to s
		in the step() method */
		PDPositions m_fExtWeighted;
		PDPositions m_fGravWeighted;
		PDScalar m_frictionCoeff;
		PDScalar m_repulsionCoeff;

		// Lists of constraints
		/* ALL constraints in the system */
		std::vector<ProjDynConstraint*> m_constraints;
		/* All strain constraints */
		std::vector<ProjDynConstraint*> m_strainConstraints;
		/* All spring constraints */
		std::vector<ProjDynConstraint*> m_springConstraints;
		/* All bending constraints */
		std::vector<ProjDynConstraint*> m_bendingConstraints;
		/* All collision constraints */
		std::vector<ProjDynConstraint*> m_collisionConstraints;
		/* All tetrahedral strain constraints */
		std::vector<ProjDynConstraint*> m_tetStrainConstraints;
		/* All handle constraints */
		std::vector<CenterConstraint*> m_handleConstraints;

		PDSparseMatrix m_lhsMatrix;
		PDSparseMatrix m_lhsCompareMat;
		PDSparseSolver m_lhsCompareSolver;
		PDVector m_rhsMasses;
		PDSparseMatrix m_massMatrix;
		PDVector m_vertexMasses;
		PDScalar m_totalMass;
		PDScalar m_timeStep;
		bool m_isSetup;
		PDSparseSolver m_linearSolver;
		Eigen::LLT<PDMatrix> m_denseSolver;
		void recomputeWeightedForces();
		StopWatch m_precomputationStopWatch;
		StopWatch m_localStepStopWatch;
		StopWatch m_globalStepStopWatch;
		StopWatch m_totalStopWatch;
		StopWatch m_localStepOnlyProjectStopWatch;
		StopWatch m_localStepRestStopWatch;
		StopWatch m_surroundingBlockStopWatch;
		StopWatch m_updatingVPosStopWatch;
		StopWatch m_constraintSummationStopWatch;
		StopWatch m_momentumStopWatch;
		StopWatch m_multiplicationForPosUpdate;
		StopWatch m_sortingForPosUpdate;
		StopWatch m_fullUpdateStopWatch;
		int m_numIterations;
		int m_numRefactorizations;
		bool once;
		bool m_recomputeFactorization;

		PDPositions m_positionCorrectionsUsedVs;
		PDPositions m_positionCorrections;
		PDPositions m_velocitiesUsedVs;
		bool m_collisionCorrection;
		PDScalar m_rayleighDampingAlpha;

		PDPositions m_initialPos;
		PDPositions m_initialPosSub;

		std::vector<unsigned int> m_grippedVertices;
		PDPositions m_grippedVertexPos;
		int* m_usedVertexMap;

		std::vector< unsigned int > m_additionalUsedVertices;

		void createPositionSubspace(unsigned int numSamples, bool useSkinningSpace = true, bool useDeimSpace = false);
		PDMatrix createSkinningWeights(unsigned int numSamples, PDScalar r);
		bool m_usingSkinSubspaces;
		
		bool m_usingPosSubspaces;   // if we do reduction for position (use subspaces for positions)
		
		/* Snapshots bases parameters */
		
		PDSparseMatrix m_massMatrixInv;
		bool m_usingPODPosSubspaces = false;
		bool m_usingSPLOCSPosSubspaces = false;
		bool m_usePosSnapBases = false;
		int m_numPosPODModes = 0;
		int m_numPosSPLOCSModes = 0;
		int m_numPosSnapBasesModes = 0;

		
		std::string m_PCABasesDir = "";
		std::string m_SPLOCSBasesDir = "";

		bool isPosSnapBasesOrtho = false;
		bool isPosSnapBasesLocal = false;
		bool isPosSnapBasesSparse = false;
		
		std::vector<PDMatrix> m_basesFunctions;
		std::vector<PDMatrixRM> m_basesFunctionsT;
		std::vector<PDMatrix> m_basesFunctionsSquared;
		std::vector<PDSparseMatrix> m_basesFunctionsSparse;
		std::vector<PDSparseMatrixRM> m_basesFunctionsTSparse;

		std::vector<PDMatrix> m_projectedLHS_mom;
		std::vector<PDMatrix> m_projectedLHS_inner;
		std::vector<PDMatrix> m_projectedRHS_mom;
		std::vector<PDMatrix> m_projectedRHS_mom_pre;

		std::vector<PDMatrix> m_projectedlhsMatrix;
		std::vector<PDSparseMatrix> m_projectedlhsMatrixSparse;

		std::vector<PDSparseMatrix> m_projectedRHS_momSparse;
		std::vector<PDSparseMatrix> m_projectedRHS_mom_preSparse;



		void finalizeSnapBasesFunctions(); 
		
		Eigen::LLT<PDMatrix> m_subspaceXSolver;
		Eigen::LLT<PDMatrix> m_subspaceYSolver;
		Eigen::LLT<PDMatrix> m_subspaceZSolver;
		
		PDSparseSolver m_subspaceXSparseSolver;
		PDSparseSolver m_subspaceYSparseSolver;
		PDSparseSolver m_subspaceZSparseSolver;
		
		Eigen::LLT<PDMatrix> m_fullSpaceXSolver;
		Eigen::LLT<PDMatrix> m_fullSpaceYSolver;
		Eigen::LLT<PDMatrix> m_fullSpaceZSolver;
		
		void projectToPODSubspace(PDPositions& b, PDPositions& x, bool isOrthogonal);
		void projectToSparsePODSubspace(PDPositions& subPos, PDPositions& fullPos, bool isBasisOrthogonal);
		
		PDSparseMatrix m_rhsXFirstTermMatrixPreSparse;
		
		
		PDSparseMatrix m_rhsYFirstTermMatrixPreSparse;
		PDSparseMatrix m_rhsZFirstTermMatrixPreSparse;
		
		
		
		PDSparseMatrix m_lhsXMatrixSampledSparse;
		PDSparseMatrix m_lhsYMatrixSampledSparse;
		PDSparseMatrix m_lhsZMatrixSampledSparse;
		
		Eigen::LLT<PDMatrix> m_denseXSolver;
		Eigen::LLT<PDMatrix> m_denseYSolver;
		Eigen::LLT<PDMatrix> m_denseZSolver;
		
		PDSparseSolver m_XSolverSparse;
		PDSparseSolver m_YSolverSparse;
		PDSparseSolver m_ZSolverSparse;
		
		PDSparseSolver m_subspaceXSystemSolverSparse;
		PDSparseSolver m_subspaceYSystemSolverSparse;
		PDSparseSolver m_subspaceZSystemSolverSparse;
		
		
		void updatePODPositionsSampling(PDPositions& fullPos, PDPositions& subPos, bool localSupport);
		
		std::vector< PDSparseMatrixRM > m_baseXFunctionsSparseBlocks;
		std::vector< PDSparseMatrixRM > m_baseYFunctionsSparseBlocks;
		std::vector< PDSparseMatrixRM > m_baseZFunctionsSparseBlocks;
		
		std::vector< PDPositions > m_additionalXConstraintsAuxTemp;
		std::vector< PDPositions > m_additionalYConstraintsAuxTemp;
		std::vector< PDPositions > m_additionalZConstraintsAuxTemp;
		
		PDMatrixRM m_baseXFunctionsSquaredBig;
		PDMatrixRM m_baseYFunctionsSquaredBig;
		PDMatrixRM m_baseZFunctionsSquaredBig;
		
		PDPositions m_positionsPODSubspace;
		PDPositions m_velocitiesPODSubspace;
		
		PDPositions m_PODpositionCorrections;
		// for random plane collision
		void resolvePlaneCollisopn(unsigned int v, PDPositions & pos, PDPositions & posCorrect);
		//void addRandomFloor(PDScalar a, PDScalar b, PDScalar c, PDScalar d, PDScalar floorCollisionWeight);
		PDScalar coeffX, coeffY, coeffZ, planeScalar;
		bool m_planeBounceCorrection;
		
		
		bool podUsedVerticesOnly = false;
		
		PDMatrix m_usedVertexXInterpolatorRHSMatrix;
		PDMatrix m_usedVertexYInterpolatorRHSMatrix;
		PDMatrix m_usedVertexZInterpolatorRHSMatrix;
		
		Eigen::LLT<PDMatrix> m_usedVertexXInterpolator;
		Eigen::LLT<PDMatrix> m_usedVertexYInterpolator;
		Eigen::LLT<PDMatrix> m_usedVertexZInterpolator;
		
		PDSparseMatrix m_usedVertexXInterpolatorRHSMatrixSparse;
		PDSparseMatrix m_usedVertexYInterpolatorRHSMatrixSparse;
		PDSparseMatrix m_usedVertexZInterpolatorRHSMatrixSparse;
		
		PDSparseSolver m_usedVertexXInterpolatorSparse;
		PDSparseSolver m_usedVertexYInterpolatorSparse;
		PDSparseSolver m_usedVertexZInterpolatorSparse;
		
		std::vector< std::vector<PDScalar> > m_usedVerticesXBase;
		std::vector< std::vector<PDScalar> > m_usedVerticesYBase;
		std::vector< std::vector<PDScalar> > m_usedVerticesZBase;
		
		std::vector< std::vector< unsigned int > > m_usedVerticesXBaseNNZ;
		std::vector< std::vector< unsigned int > > m_usedVerticesYBaseNNZ;
		std::vector< std::vector< unsigned int > > m_usedVerticesZBaseNNZ;
		// --------------------------
		// for QDEIM case
		bool m_usingQDEIMComponents= false;
		
		int m_numQDEIMModes;
		
		void loadQDEIMnonlinearSubspace(int numQDEIMModes, bool useSkinningSpace, bool usePositionPODSpace, bool rhsInterpolation);
		PDMatrix tempMqdeim;
		PDMatrix tempSqdeim, qdeimBlocks;
		//Eigen::Matrix< unsigned int, -1, 1 > qdeimBlocks;
		
		PDMatrix m_xMqdeim;
		PDMatrix m_yMqdeim;
		PDMatrix m_zMqdeim;
		
		int deimBasisCols;
		
		std::vector<PDMatrix> m_Mqdeim;
		std::vector<std::vector< unsigned int >> m_Sqdeim;
		
		std::vector< unsigned int >  m_SqdeimX, m_SqdeimY, m_SqdeimZ;
		
		PDSparseMatrix m_xMqdeimSparse;
		PDSparseMatrix m_yMqdeimSparse;
		PDSparseMatrix m_zMqdeimSparse;
		
		std::vector< PDSparseMatrix > qdeimSTMx, qdeimSTMy, qdeimSTMz;
		std::vector< PDSparseMatrix > UTSTx, UTSTy, UTSTz;
		
		PDMatrix UTSTMx, UTSTMy, UTSTMz;
		PDSparseMatrix UTSTMxSparse, UTSTMySparse, UTSTMzSparse;

		PDMatrix STMx, STMy, STMz;
		
		int p_constaintSize;
		
		bool m_TetStrainOnly;
		
		std::vector< unsigned int > m_constraintQDEIMVertexSamples;
		std::vector< unsigned int > m_constraintQDEIMTriSamples;
		std::vector< unsigned int > m_constraintQDEIMTetSamples;
		
		
		void createQDEIMConstraintTetStrainSampling();
		std::vector< std::vector< unsigned int > > m_vertsPerTet;
		void initQDEIMRHSInterpolGroup(RHSInterpolationGroup& g, std::vector<unsigned int>& samples, std::vector<PDMatrix>& qdeimBasis);
		
		PDMatrix nonlinearSnapshots;
		//PDMatrix STMx, UTSTy, UTSTz;
		bool m_solveDeimLS = false;
		//---------------------------
		// for only rhsInrepolation case
		PDSparseSolver m_linearFullLHSinterploRHSSolver;
		PDPositions m_rhsInterpol;
		//---------------------------
		// Quaternions case----------
		void projectQuaternionsToSubspace(Eigen::MatrixXd& b, PDPositions& x, bool isBasisOrthogonal);
		//---------------------------
		// for recording constaints snapshots for QDEIM
		bool recordingSTpSnapshots = false;
		bool recordingPSnapshots = false;
		bool recordingTetStrainOnly = false;
				
				
				
		std::vector< unsigned int > m_samples;
		PDMatrix m_baseFunctions;
		PDMatrix m_snapshotsBasesTmp;
		PDMatrix m_baseFunctionWeights;
		PDMatrix m_baseFunctionsTransposed;
		PDMatrix m_baseFunctionsSquared;
		PDMatrix m_rhsFirstTermMatrix;
		PDPositions m_fExtWeightedSubspace;
		PDPositions m_fGravWeightedSubspace;
		PDScalar m_baseFunctionRadius;
		PDMatrix m_lhsMatrixSampled;
		PDMatrix m_lhsMatrixSampledStiff;
		PDMatrix m_rhsCollisionTerm;
		PDMatrix m_rhsStabTerm;
		PDPositions m_positionsSubspace;
		PDPositions m_positionsUsedVs;
		PDPositions m_velocitiesSubspace;
		Eigen::LLT<PDMatrix> m_subspaceSolver;
		Eigen::LLT<PDMatrix> m_usedVertexInterpolator;
		PDMatrix m_usedVertexInterpolatorRHSMatrix;
		void projectToSubspace(PDPositions& b, PDPositions& x, bool isOrthogonal);
		
		PDSparseMatrix m_rhsFirstTermMatrixSparse;
		PDSparseMatrix m_baseFunctionsTransposedSparse;
		PDSparseMatrixRM m_baseFunctionsSparse;
		std::vector< PDSparseMatrixRM > m_baseFunctionsSparseBlocks;
		PDSparseMatrix m_usedVertexInterpolatorRHSMatrixSparse;
		PDSparseSolver m_subspaceSystemSolverSparse;
		PDSparseSolver m_usedVertexInterpolatorSparse;

		PDMatrix m_subspaceLHS_mom;
		PDMatrix m_subspaceLHS_inner;
		PDMatrix m_rhsFirstTermMatrixPre;

		bool* m_collidedVerts;

		void finalizeBaseFunctions();

		PDPositions m_rhs;

		PDPositions rhs;
		PDPositions rhs2;

		PDScalar m_stiffnessFactor = 1.;

		int m_numSamplesPosSubspace;

		// RHS Interpolation stuff
		int m_numConstraintSamples;
		std::vector<unsigned int> m_allVerts;
		void createConstraintSampling(unsigned int numSamples);
		/* List of vertex indices at which constraints are being sampled. I.e. if a constraint contains
		this vertex, it is evaluated. Note that this means that there are more vertices involved in
		the sampled constraints than just the ones in this list, which is where the list m_usedVertices
		comes in. */
		std::vector< unsigned int > m_constraintVertexSamples;
		/* List of triangle indices at which constraints are being sampled. For each vertex, one adjacent
		triangle is being selected (basically arbitrarily). */
		std::vector< unsigned int > m_constraintTriSamples;
		/* List of tetrahedron indices at which constraints are being sampled. For each vertex, one adjacent
		tetrahedron is being selected (basically arbitrarily). */
		std::vector< unsigned int > m_constraintTetSamples;
		/* List of edge indices at which constraints are being sampled. For each vertex, one adjacent
		edge is being selected (basically arbitrarily). */
		std::vector< unsigned int > m_constraintEdgeSamples;
		/* When using constraint sampling, this list contains all vertices that are involved in
		the sampled constraints. This is different from the list m_constraintSamples,
		which contains only the vertices at which constraints were sampled.
		MOST IMPORTANTLY this list contains the non-zero rows of the sum on the r.h.s of the
		linear system.*/
		std::vector< unsigned int > m_usedVertices;
		/* The list of constraints that are being sampled. 
		   These get chosen by the rhs interpolation groups*/
		std::vector< ProjDynConstraint* > m_sampledConstraints;
		bool m_constraintSamplesChanged;

		/* Resolves collisions for the vertex with index v in the vector pos and writes the
		   correction vector into row v of posCorrect*/
		void resolveCollision(unsigned int v, PDPositions& pos, PDPositions& posCorrect);

		/* Vertices at which collisions are being resolved, required to build list of "used vertices" */
		//std::vector< unsigned int > m_collisionSamples;

		std::vector< PDPositions > m_additionalConstraintsAuxTemp;
		

		/* Updates positions at all vertices used by sampled constraints */		
		void updatePositionsSampling(PDPositions& fullPos, PDPositions& subPos, bool usedVerticesOnly);
		void evaluatePositionsAtUsedVertices(PDPositions& fullPos, PDPositions& subPos);
		std::vector< std::vector<PDScalar> > m_usedVerticesBase;
		std::vector< std::vector< unsigned int > > m_usedVerticesBaseNNZ;

		/* Updates the list of used vertices (NOT the positions at used vertices) */
		void updateUsedVertices();

		/* Whether we only evaluate a few samples of the constraints and interpolate
		   the full rhs of the global step using some subspace for the rhs. */
		bool m_rhsInterpolation;
		void addAdditionalConstraints(PDPositions& pos, PDPositions& rhs, bool* collidedVertices);
		std::vector<ProjDynConstraint*> m_additionalConstraints;
		std::vector<RHSInterpolationGroup> m_snapshotGroups;

		void initRHSInterpolGroup(RHSInterpolationGroup& g, std::vector<unsigned int>& samples); /// , PDMatrix& hessian);
		int m_rhsInterpolBaseSize;

		PDScalar m_rhsRegularizationWeight;
		PDScalar m_rhsInterpolWeightRadiusMultiplier;
		PDMatrix m_rhsInterpolReusableWeights;

		/* Count of frames (full simulation steps) since the start of the simulation */
		//int m_frameCount;


		PDScalar m_lastRHSNorm = 0;

		// CUDA stuff
#ifdef PROJ_DYN_USE_CUBLAS
		CUDAMatrixVectorMultiplier* m_usedVertexUpdater;
		CUDAMatrixVectorMultiplier* m_rhsEvaluator;
		CUSparseMatrixVectorMultiplier* m_usedVertexUpdaterSparse;
		PDMatrix m_projUsedVerts;
		PDMatrix m_rhsEvalMat;
		PDVector m_curTempVec;
#endif
	};

	struct SparseEntry {
		unsigned int i, j;
		double entry;
	};

}
