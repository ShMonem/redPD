
 To reproduce results in *Improved-Projective-Dynamics-Global-Using-Snapshots-based-Reduced-Bases* SIGGRAPH23 [1st place student competition award-winning paper](https://dl.acm.org/doi/10.1145/3588028.3603665).

 1. Clone this repository 									
	``` 
	git clone https://github.com/ShMonem/redPD.git
	```
2. Build the project using your favorate software, or using terminal as below:
	```
   cd redPD &&
   mkdir build && cd build 
   cmake ..
   make
    ```
3. Full order simulations of a `bunny.obj` mesh falling under gravitational force will appear on your screen.
  - Press key "1" to see some statistics about running time.
  - Two directories will be created
	- `redPD/results/bunny/_gravitationalFall/position_snapshots/FOM/` stores <.off> snapshots files for FOM simulations.
  	- `redPD/results/bunny/_gravitationalFall/png/FOM` stores <.png> fot the same snapshots.  
  - The options of storing the above files can be changed using `macros` defined in `ProjDynSimolator.h`
	```
	// Setting results storage macros
	// If storing frames in .png format (e.g. for numerical stability comparision)
	#define STORE_FRAMES_PNG true
	// If storing frames in .off format (e.g. for FOM snapshots collection, or error computation between different reduction methods)
	#define STORE_FRAMES_OFF true
	```
4. Clone [animSnapBases](https://github.com/ShMonem/animSnapBases) that containes the python code for bases computing
	```
	git clone https://github.com/ShMonem/animSnapBases.git
	```
5. Copy `redPD/results/bunny/_gravitationalFall/position_snapshots/FOM/` to `animSnapBases/input_data/bunny/_gravitationalFall/position_snapshots/FOM`,
6. Choose the bases you want to compute as explained in the [<README.dev.md>](https://github.com/ShMonem/animSnapBases/blob/main/config/README.dev.md),
  - Two directories will be created (*ex:* PCA, using rigid aligned snapshots input 200 snapshots frames):
	- `animSnapBases/results/bunny/_gravitationalFall/q_snapshots_h5/aligned_snapshots200outOf200_Frames_1_increment__alignedRigid.h5` stores `.h5` file for snapshots viualization.
	- `animSnapBases/results/bunny/_gravitationalFall/q_bases/PCA_alignedRigid_Volkwein_Standarized_Local_nonOrthogonalized_Release/200outOf200_Frames_/1_increment_200_alignedRigid_bases` stores `.bin` files for bases
7. Copy the latest folder containing the bases to `redPD/bases/bunny/_gravitationalFall/q_bases/PCA_alignedRigid_Volkwein_Standarized_Local_nonOrthogonalized_Release/200outOf200_Frames_/1_increment_200_alignedRigid_bases`
  _ It contains `.bin` for different number of bases `in range (10, 200, 10)`, hence any of these `.bin` can be red by the `c++` code 
8. In `redPD/src/main.cpp`, choose number of PCA bases used, for instance
	```
	numberPositionPCAModes = 200
    ```
9. Run simulation again 
10. Notice time acceleration for the global step computation, when key "1" is pressed during simlations.
  - Note: `.png` and `.off` will also be stored for the reduced simulations as decribed in point 3.