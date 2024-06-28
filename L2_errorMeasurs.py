"""
ShaimaaMonem 29.12.21
Test running libigl and polyscope together
"""
import struct
import sys
#sys.path.append("/home/shaimaa/anaconda3/lib/python3.7/site-packages")  #  where "meshplot is"

import scipy as sp
import numpy as np
from numpy import linalg as npla
import scipy.sparse
import scipy.sparse.linalg
import csv 
#---------------------------------
# # polyscope and potpourri3d includes
import polyscope as ps
import potpourri3d as pp3d
#---------------------------------
# # libigl includes
import igl
#import meshplot as mshp
#from meshplot import plot, subplot, interact
import os
root_folder = os.getcwd()
# #root_folder = os.path.join(os.getcwd(), "tutorial")
## Load a mesh in OFF format
v, f = igl.read_triangle_mesh("h5View.obj")
## Print the vertices and faces matrices
print("Vertices: ", len(v))
print("Faces: ", len(f))
GauCur = igl.gaussian_curvature(v, f)
print("GaussianCurvature: ", len(GauCur))  # per vertex scalar
#mshp.plot(v, f, k, return_plot=True)    # doesn't show any plots
#---------------------------------
# # Polyscope and Potpourri3d
# enable auto centering and scaling
ps.set_autocenter_structures(True)
ps.set_autoscale_structures(True)
# set the camera pose explicitly

# Initialize polyscope
ps.init()
#(V, F)= pp3d.read_mesh("armadillo.obj")
(V, F)= pp3d.read_mesh("bunny.obj")
(V1, F1)= pp3d.read_mesh("armadillo.obj")
(V2, F2)= pp3d.read_mesh("h5View.obj")
print(V2.shape)
## = Mesh test
#V, F = pp3d.read_mesh("bunny_small.ply")
#ps_mesh = ps.register_surface_mesh("mesh", V, F)     # --> the original bunny

#ps_mesh2 = ps.register_surface_mesh("mesh2", V2, F2)
#ps_mesh1 = ps.register_surface_mesh("mesh1", V1, F1)  ## polyscope can show many meshes at same time

numFrames = 200

accumVertexErrorSkin = np.zeros((V.shape[0]))

accumVertexErrorSPLOCS = np.zeros((V.shape[0]))
"""
k=80
for f in range(numFrames):
	(VFull, FFull)= \
	pp3d.read_mesh("/home/shaimaa/libigl/tutorial/doubleHRPD/meshFrames_fullSimulationsOFF/bunny/fullPositionsFewFrames/m_bunnyPositions_"+str(f)+".off")
	
	(VPPOD, FPPOD)= \
	pp3d.read_mesh("/home/shaimaa/libigl/tutorial/doubleHRPD/meshFrames_fullSimulationsOFF/bunny/fullPODPosNoRigidAlignFrames/fullPODPosF601K"+str(k)+"/subSpacePos"+str(f)+".off")
	LfPOD = []
	
	
	(VPCSkin, FPCSkin)= \
	pp3d.read_mesh("/home/shaimaa/libigl/tutorial/doubleHRPD/meshFrames_fullSimulationsOFF/bunny/fullPODPositionsFewFrames/m_bunnyPositions_"+str(f)+".off")
	LfSkin = []
	
	(VPSPLOCS, FPSPLOCS)= \
	pp3d.read_mesh("/home/shaimaa/libigl/tutorial/doubleHRPD/meshFrames_fullSimulationsOFF/bunny/fullSPLOCSPositionsFewFrames/m_bunnyPositions_"+str(f)+".off")
	LfSPLOCS = []
	
	# For a frame f, Lf contains error at each vertex
	for i in range(VFull.shape[0]):
		LfSkin.append(npla.norm(VFull[i][:]-VPCSkin[i][:]) )
		LfPOD.append(npla.norm(VFull[i][:]-VPPOD[i][:]) )
		LfSPLOCS.append(npla.norm(VFull[i][:]-VPSPLOCS[i][:]) )
	
	# Accumulate errors for all frames
	accumVertexErrorSkin += np.array(LfSkin)
	accumVertexErrorPOD += np.array(LfPOD)
	accumVertexErrorSPLOCS += np.array(LfSPLOCS)
	
	
	skinL2.append(LfSkin) 
	podL2.append(LfPOD)
	splocsL2.append(LfSPLOCS)
	#print(splocsL2)
	
skinL2 /= np.sqrt(numFrames*V.shape[0]) 
podL2 /= np.sqrt(numFrames*V.shape[0])  
splocsL2 /= np.sqrt(numFrames*V.shape[0])


print("maximal L2 norm, case SkinSubspaces = %f" % np.array(skinL2).max())
print("mean L2 norm, case SkinSubspaces= %f" % np.array(skinL2).mean())

print("maximal L2 norm, case PODSubspaces = %f" % np.array(podL2).max())
print("mean L2 norm, case PODSubspaces= %f" % np.array(podL2).mean())

print("maximal L2 norm, case SPLOCSsubspaces = %f" % np.array(splocsL2).max())
print("mean L2 norm, case SPLOCSsubspaces= %f" % np.array(splocsL2).mean())
"""

	

"""
fileBasis = open ("sqrtMvertWeigUbasisVolkweinF202K200.bin", "rb")
N = struct.unpack('<i', fileBasis.read(4))[0]
K = (struct.unpack('<i', fileBasis.read(4))[0])//3
basis = np.zeros((N, K, 3)) 
for d in range(3):
	for k in range(K):
		for i in range(N):
			value = struct.unpack('<d', fileBasis.read(8))[0] # read 8 byte and interpret them as little endian double
			basis[i, k, d] = value
			#print(value)
fileBasis.close()

component0 = np.zeros((N, 3))
for d in range(3):
	for i in range(N):
		component0[i, d] = basis[i, 0, d]
#ps_mesh3 = ps.register_surface_mesh("mesh3", component0, F)

component10 = np.zeros((N, 3))
for d in range(3):
	for i in range(N):
		component10[i, d] = basis[i, 10, d]
ps_mesh10 = ps.register_surface_mesh("mesh10", component10, F)
"""
#---------------------------------
#create a file and write errorrs results into it

skinL2 = []
podL2 = []
splocsL2 = []
def callback():
	headerComp = ['numComponent', 'Frame','minError', 'mnaxError','maxAccumError', 'minAccumError', 'sumAccumError']
	
	with open('PODerrorsVal.csv', 'w', encoding='UTF8') as errorsPODFile:
		writer = csv.writer(errorsPODFile)
		writer.writerow(headerComp)
		
		print("Case POD basis:")
		print("Components     | min error over all frames|  max error over all frames | min accumulated error  | max accumulated error | sum accumulated error")
		for k in range(10,201, 10):
			accumVertexErrorPOD = np.zeros((V2.shape[0]))
			'''
			# cearte directory to store the screenshots  # uncomment to safe snapshots	
			shotsDirectory = "fullPODPosF601K"+str(k)
			storeFoulder = "/home/shaimaa/libigl/tutorial/doubleHRPD/errorScreenShots/fullpodF601K"
			path = os.path.join(storeFoulder, shotsDirectory)
			os.mkdir(path)
			'''
			# Executed every frame
			for count in range(numFrames+1):
				if count < numFrames:
					(VFull, FFull)= \
					pp3d.read_mesh("/home/shaimaa/libigl/tutorial/doubleHRPD/meshFrames_fullSimulationsOFF/bunny/fullPositionsFewFrames/m_bunnyPositions_"+str(count)+".off")

					(VPPOD, FPPOD)= \
					pp3d.read_mesh("/home/shaimaa/libigl/tutorial/doubleHRPD/meshFrames_fullSimulationsOFF/bunny/fullPODPosNoRigidAlignFrames/fullPODPosF601K"+str(k)+"/subSpacePos"+str(count)+".off")
					LfPOD = []
					ps_mesh = ps.register_surface_mesh("mesh", VPPOD, FPPOD)
					
					# For a frame count, Lf contains error at each vertex
					for i in range(VFull.shape[0]):
						LfPOD.append(npla.norm(VFull[i][:]-VPPOD[i][:]))    # kavan error
					
					frameError = np.array(LfPOD)/(np.sqrt(numFrames*V.shape[0]))
					ps_mesh.add_scalar_quantity("Error", frameError, defined_on='vertices', cmap='jet')
					podL2.append(frameError)
					#print(count)
						
				'''
				# view first snapshot and adjust it
				# then safe screenshots for all frames at each fixed component
				# uncomment to safe snapshots 			
				if count == 0:
					ps.show()
				'''	        	
				accumVertexErrorPOD += frameError
				
				
					
				if count == numFrames:
				# at the end show only accumulative error on the original mesh 
					ps_mesh2 = ps.register_surface_mesh("mesh", V2, F2)	  
					ps_mesh2.add_scalar_quantity("Error", accumVertexErrorPOD, defined_on='vertices', cmap='jet')
					#ps.show()
				'''	
				# save an image with accumulated error
				ps.screenshot("/home/shaimaa/libigl/tutorial/doubleHRPD/errorScreenShots/fullpodF601K/fullPODPosF601K"+str(k)+"/podErrors_"+str(count)+".png")
                               '''
                               
			errorsList = [k, count, np.array(podL2).min(), np.array(podL2).max(), accumVertexErrorPOD.max(), accumVertexErrorPOD.min(), accumVertexErrorPOD.sum()]
			writer.writerow(errorsList)
			print(k, "		 ", np.array(podL2).min(), np.array(podL2).max(), accumVertexErrorPOD.min(), accumVertexErrorPOD.max(), accumVertexErrorPOD.sum())

	errorsPODFile.close()		
	
	
	with open('LBSerrorsVal.csv', 'w', encoding='UTF8') as errorsLBSFile:
		writer = csv.writer(errorsLBSFile)
		writer.writerow(headerComp)
		
		print("Case LBS basis:")
		print("Components     | min error over all frames|  max error over all frames | min accumulated error  | max accumulated error | sum accumulated error")
		for k in range(180,201, 10):
			accumVertexErrorPOD = np.zeros((V2.shape[0]))
			
			# cearte directory to store the screenshots  # uncomment to safe snapshots	
			shotsDirectory = "fullLBSPosF601K"+str(k)
			storeFoulder = "/home/shaimaa/libigl/tutorial/doubleHRPD/errorScreenShots/fulllbsF601K"
			path = os.path.join(storeFoulder, shotsDirectory)
			os.mkdir(path)
			
			# Executed every frame
			for count in range(numFrames+1):
				if count < numFrames:
					(VFull, FFull)= \
					pp3d.read_mesh("/home/shaimaa/libigl/tutorial/doubleHRPD/meshFrames_fullSimulationsOFF/bunny/fullPositionsFewFrames/m_bunnyPositions_"+str(count)+".off")

					(VPPOD, FPPOD)= \
					pp3d.read_mesh("/home/shaimaa/libigl/tutorial/doubleHRPD/meshFrames_fullSimulationsOFF/bunny/fullLBSOnlyPosFrames/LBS"+str(k)+"/subSpacePos"+str(count)+".off")
					LfPOD = []
					ps_mesh = ps.register_surface_mesh("mesh", VPPOD, FPPOD)
					
					# For a frame count, Lf contains error at each vertex
					for i in range(VFull.shape[0]):
						LfPOD.append(npla.norm(VFull[i][:]-VPPOD[i][:]) / npla.norm(VFull[i][:]))
					
					frameError = np.array(LfPOD)/(np.sqrt(numFrames*V.shape[0]))
					ps_mesh.add_scalar_quantity("Error", frameError, defined_on='vertices', cmap='jet')
					skinL2.append(frameError)
					#print(count)
						
				
				# view first snapshot and adjust it
				# then safe screenshots for all frames at each fixed component
				# uncomment to safe snapshots 			
				if count == 0:
					ps.show()
					        	
				accumVertexErrorPOD += frameError
				
				
					
				if count == numFrames:
				# at the end show only accumulative error on the original mesh 
					ps_mesh2 = ps.register_surface_mesh("mesh", V2, F2)	  
					ps_mesh2.add_scalar_quantity("Error", accumVertexErrorPOD, defined_on='vertices', cmap='jet')
					#ps.show()
					# save an image with accumulated error
				ps.screenshot("/home/shaimaa/libigl/tutorial/doubleHRPD/errorScreenShots/fulllbsF601K/fullLBSPosF601K"+str(k)+"/podErrors_"+str(count)+".png")

			errorsList = [k, count, np.array(skinL2).min(), np.array(skinL2).max(), accumVertexErrorPOD.max(), accumVertexErrorPOD.min(), accumVertexErrorPOD.sum()]
			writer.writerow(errorsList)
			print(k, "		 ", np.array(skinL2).min(), np.array(skinL2).max(), accumVertexErrorPOD.min(), accumVertexErrorPOD.max(), accumVertexErrorPOD.sum())
			
	errorsLBSFile.close()		
	
	with open('SPLOCSerrorsVal.csv', 'w', encoding='UTF8') as errorsSPLOCSFile:
		writer = csv.writer(errorsSPLOCSFile)
		writer.writerow(headerComp)
		
		print("Case SPLOCS basis:")
		print("Components     | min error over all frames|  max error over all frames | min accumulated error  | max accumulated error  | sum accumulated error")
		for k in (4,6,8,10,20,30,40,50,60,70):
			accumVertexErrorPOD = np.zeros((V2.shape[0]))
			
			# cearte directory to store the screenshots  # uncomment to safe snapshots	
			shotsDirectory = "fullSPLOCSPosF601K"+str(k)
			storeFoulder = "/home/shaimaa/libigl/tutorial/doubleHRPD/errorScreenShots/fullsplocsF601K"
			path = os.path.join(storeFoulder, shotsDirectory)
			os.mkdir(path)
			
			# Executed every frame
			for count in range(numFrames+1):
				if count < numFrames:
					(VFull, FFull)= \
					pp3d.read_mesh("/home/shaimaa/libigl/tutorial/doubleHRPD/meshFrames_fullSimulationsOFF/bunny/fullPositionsFewFrames/m_bunnyPositions_"+str(count)+".off")

					(VPPOD, FPPOD)= \
					pp3d.read_mesh("/home/shaimaa/libigl/tutorial/doubleHRPD/meshFrames_fullSimulationsOFF/bunny/fullSPLOCSPosNotRigidAlignFrames/fullSPLOCSPosF601K"+str(k)+"/subSpacePos"+str(count)+".off")
					LfPOD = []
					ps_mesh = ps.register_surface_mesh("mesh", VPPOD, FPPOD)
					
					# For a frame count, Lf contains error at each vertex
					for i in range(VFull.shape[0]):
						LfPOD.append(npla.norm(VFull[i][:]-VPPOD[i][:]) / npla.norm(VFull[i][:]) )
					
					frameError = np.array(LfPOD)/(np.sqrt(numFrames*V.shape[0]))
					ps_mesh.add_scalar_quantity("Error", frameError, defined_on='vertices', cmap='jet')
					splocsL2.append(frameError)
					#print(count)
						
				
				# view first snapshot and adjust it
				# then safe screenshots for all frames at each fixed component
				# uncomment to safe snapshots 			
				if count == 0:
					ps.show()
					        	
				accumVertexErrorPOD += frameError
				
				
					
				if count == numFrames:
				# at the end show only accumulative error on the original mesh 
					ps_mesh2 = ps.register_surface_mesh("mesh", V2, F2)	  
					ps_mesh2.add_scalar_quantity("Error", accumVertexErrorPOD, defined_on='vertices', cmap='jet')
					#ps.show()
					# save an image with accumulated error
				ps.screenshot("/home/shaimaa/libigl/tutorial/doubleHRPD/errorScreenShots/fullsplocsF601K/fullSPLOCSPosF601K"+str(k)+"/podErrors_"+str(count)+".png")
				
			errorsList = [k, count, np.array(splocsL2).min(), np.array(splocsL2).max(), accumVertexErrorPOD.max(), accumVertexErrorPOD.min(),accumVertexErrorPOD.sum()]
			writer.writerow(errorsList)
			print(k, "		 ", np.array(splocsL2).min(), np.array(splocsL2).max(), accumVertexErrorPOD.min(), accumVertexErrorPOD.max(), accumVertexErrorPOD.sum())
				
	
	errorsSPLOCSFile.close()		



ps.set_user_callback(callback)
ps.set_always_redraw(True)	
ps.show()
### Register a mesh
ps.clear_user_callback()




"""
Kvan error: shows the best errors reanges so far
Vertices:  14290
Faces:  28576
GaussianCurvature:  14290
[polyscope] Backend: openGL3_glfw -- Loaded openGL version: 4.6 (Core Profile) Mesa 21.2.6
(14290, 3)
Case POD basis:
Components     | min error over all frames|  max error over all frames | min accumulated error  | max accumulated error  | sum accumulated error
10 		  2.568358563576087e-09 0.0015558810345612073 0.017105772693890447 0.2201710068263542 1064.4980356194076
20 		  3.9472637866767704e-11 0.0016417120380261558 0.014982995940936524 0.189513310125666 992.2420960404002
30 		  3.9472637866767704e-11 0.0016417120380261558 0.016078035469930002 0.182741247246954 979.5955260466181
40 		  3.7743471346784793e-11 0.0017301806351797643 0.018297671645777145 0.19833927422976708 1085.8440347497087
50 		  1.6367499800321788e-11 0.0017917041806610704 0.025105614145483488 0.2050354513953923 1197.7742007039888
60 		  1.6367499800321788e-11 0.0017917041806610704 0.031055639840474037 0.19357124336357562 1285.642069719359
70 		  1.6367499800321788e-11 0.0017917041806610704 0.029726515795467787 0.19298821763309318 1254.9099106971062
80 		  1.6367499800321788e-11 0.0017917041806610704 0.03495803361054369 0.09992738074105667 740.0126596443876
90 		  1.6367499800321788e-11 0.0017917041806610704 0.03659296949617627 0.10006338903549838 747.897887345139
100 		  1.6367499800321788e-11 0.0017917041806610704 0.03945755555120851 0.10007952201281732 752.8229663087147
110 		  1.1964056379607695e-11 0.0017917041806610704 0.04389304233647975 0.10329059393060634 786.8257229706517
120 		  1.1416657720143327e-11 0.0017917041806610704 0.04208325968915272 0.09711472520949142 754.8701780122308
130 		  1.1416657720143327e-11 0.0017917041806610704 0.04299712253892655 0.10378046092516292 740.6937204795718
140 		  1.1416657720143327e-11 0.0017917041806610704 0.04093454952700053 0.09725593702864313 722.2332221808005
150 		  6.348798135057786e-12 0.0017917041806610704 0.04088752552887579 0.09797016371679781 724.7208320378061
160 		  6.348798135057786e-12 0.0017917041806610704 0.04109860682419811 0.0948679288136849 722.435372988901
170 		  6.348798135057786e-12 0.0017917041806610704 0.03930436036797975 0.11069644820339263 725.7191266545279
180 		  6.348798135057786e-12 0.0017917041806610704 0.035138298683843334 0.11329772671125332 723.4630224382569
190 		  6.348798135057786e-12 0.0017917041806610704 0.03392203455757392 0.11069683028917089 714.2307277338684
200 		  6.348798135057786e-12 0.0017917041806610704 0.03352896040917374 0.10632523987429576 707.0663661295505
Case LBS basis:
Components     | min error over all frames|  max error over all frames | min accumulated error  | max accumulated error 
10 		  2.6239946715382658e-06 inf inf inf inf
20 		  2.6239946715382658e-06 inf inf inf inf
30 		  9.701571195832102e-07 inf 0.033560846374703204 0.2364351382431172 1409.9757389304873
40 		  9.701571195832102e-07 inf inf inf inf
50 		  5.306448819405344e-07 inf 0.03280539110548951 0.24415778159719037 1489.1063372266985
60 		  5.306448819405344e-07 inf 0.031252454035505535 0.23876122165934235 1448.6518130529028
70 		  5.306448819405344e-07 inf 0.03275326482892777 0.24744783078585536 1517.3956961413858
80 		  5.306448819405344e-07 inf 2.3432231683834285e+136 8.845847600245323e+142 1.1474715524438248e+146
100 		  4.911676410907793e-07 inf 0.025831592991261774 0.22929298039721474 1435.0188401227288
110 		  4.895082390129409e-07 inf 0.026956324967066972 0.23092510061311675 1432.8461756907632
120 		  2.6753453439415794e-07 inf 0.024104415180646065 0.22799391892776122 1428.196505233317
130 		  2.6753453439415794e-07 inf 0.02364226645233669 0.21566892662717058 1383.3214158988053
140 		  2.6753453439415794e-07 inf 0.022601373477448304 0.23042330772582711 1424.5001124432401
150 		  2.6753453439415794e-07 inf 0.023110585395735152 0.2236046996893542 1380.949226120987
160 		  2.6753453439415794e-07 inf 3.446416353825867e+125 4.1721531448308714e+133 5.032170989267462e+136

180 		  5.924150219314741e-07 0.0013972281616478008 0.0362500255035808 0.17692817275974212 1141.61065003146
190 		  5.177872181525946e-07 0.0019367574556758535 0.026996757473718526 0.22355139698156537 1375.1263143845133
200 		  5.177872181525946e-07 0.0019367574556758535 0.029350968398980503 0.2233071893027228 1363.2318910670429

#========================================================================================================================================================
Relative Kavan error

Case POD basis:
Components     | min error over all frames|  max error over all frames | min accumulated error  | max accumulated error | sum accumulated error
10 		  6.862853410136273e-10 0.005894770157185211 0.01876313592731818 0.2730211697947741 1427.9106931339857
20 		  1.3635943502915062e-11 0.005894770157185211 0.016816888606566854 0.23079324787718056 1292.4740859080018
30 		  1.3635943502915062e-11 0.005894770157185211 0.0180825898653019 0.23606013367146228 1284.4403246031766
40 		  1.3635943502915062e-11 0.006252455192561762 0.019721367036365383 0.2646949760755916 1437.9030671007981
50 		  6.217341422976179e-12 0.006252455192561762 0.022652186628991537 0.26630863042501973 1564.2734699328055
60 		  6.217341422976179e-12 0.006252455192561762 0.02825581999405248 0.28802399943829027 1671.4463900579524
70 		  6.217341422976179e-12 0.006252455192561762 0.026916980120102234 0.2852884146126927 1643.8541061843518
80 		  6.217341422976179e-12 0.006252455192561762 0.029692444387458904 0.22191321152909976 1010.2131522053195
90 		  6.217341422976179e-12 0.006252455192561762 0.0312674988105072 0.23066675526765537 1016.1099742139164
100 		  6.217341422976179e-12 0.006252455192561762 0.03604833092538339 0.22161463999276915 1025.2815674942112
110 		  3.713445836316232e-12 0.006252455192561762 0.03905442558476461 0.2287032567903158 1083.8485350122287
120 		  3.713445836316232e-12 0.006252455192561762 0.03662389177189697 0.21624690587622758 1032.908642757525
130 		  3.713445836316232e-12 0.006252455192561762 0.03520065816859519 0.20686501518449746 1007.6810678111676
140 		  3.713445836316232e-12 0.006252455192561762 0.03378533088528953 0.19943903077196656 976.5346943454317
150 		  2.562089038683467e-12 0.006252455192561762 0.032568385207810566 0.19784857185904087 982.1472551611641
160 		  2.562089038683467e-12 0.006252455192561762 0.03243616920522391 0.20090169188297913 981.7404513401909
170 		  2.562089038683467e-12 0.006252455192561762 0.031290446310676776 0.200635856514388 984.0330953406927
180 		  2.562089038683467e-12 0.006252455192561762 0.027804121412021574 0.20584275703651997 984.7375640438956
190 		  2.562089038683467e-12 0.006252455192561762 0.027442167022271084 0.20234992004678617 972.0764923031958
200 		  2.562089038683467e-12 0.006252455192561762 0.026897403059001995 0.19777790541913784 960.9670865853261
Case LBS basis:
Components     | min error over all frames|  max error over all frames | min accumulated error  | max accumulated error | sum accumulated error
180 		  7.603689906884975e-07 0.005937108310259032 0.02914974399233521 0.2615138804017624 1522.8979977137024
190 		  5.221061004621827e-07 0.0071169876651875205 0.029314917948400795 0.3239352739289453 1767.771413811361
200 		  5.221061004621827e-07 0.0071169876651875205 0.03143614597781433 0.319996967286883 1751.0858931463358
Case SPLOCS basis:
Components     | min error over all frames|  max error over all frames | min accumulated error  | max accumulated error  | sum accumulated error
4 		  2.666170247712625e-10 0.006828985618848212 0.020956637779457336 0.34350243044388373 1626.2537918007401
6 		  2.666170247712625e-10 0.006864891529950646 0.02648853114213932 0.3307784991364477 1573.555058573002
8 		  2.666170247712625e-10 0.006864891529950646 0.029526659920980684 0.30754009017618567 1597.115193491954
10 		  1.4612598800793217e-10 0.006864891529950646 0.026882373141627097 0.28418855500075924 1542.7317707676707
20 		  1.4612598800793217e-10 0.006864891529950646 0.02387911162054526 0.2733524292582322 1471.678726395452
30 		  4.885358014708391e-11 0.006864891529950646 0.022935058461609238 0.261490140592416 1349.3921450998137
40 		  4.658175360771696e-11 0.006864891529950646 0.02294990402100943 0.2711068838945416 1412.9537079228642
50 		  3.546629160258118e-11 0.006864891529950646 0.022188658401032586 0.26802484522940767 1409.209765693232
60 		  3.546629160258118e-11 0.006864891529950646 0.0225401594306168 0.26441466906603567 1388.8980177311682
70 		  3.4703370571823606e-11 0.006864891529950646 0.02369073463093215 0.2626526828287453 1360.135958701048

From my prespective Kavan error measure makes more sense because we are taking the error for all frames together at each fixed vertex
while a relative error should be measured at a fixed time for all space points

"""

