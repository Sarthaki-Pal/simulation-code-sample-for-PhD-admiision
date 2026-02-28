## Requirements

- python modules : `healpy`, `numpy`, `matplotlib`, `scipy`, `astropy`, `numexpr`
- intstall the packages : `libgsl-dev`, `libfftw3-dev` and `libcfitsio-dev` by running the following command in the terminal

  ```bash
  sudo apt install libgsl-dev libfftw3-dev libcfitsio-dev
  ```

## Run the code

1. Run the `make_tge.sh` bash script in the terminal :

   ```bash
   ./make_tge.sh
   ```
   This will create `tge` executable.
2. Change the amplitude and beta in the `params_sim.py` in the `./simulations` directory : set `blt.Amp` and `blt.beta` to your required values. Run the `simulation.sh` bash script in the terminal with input fits file :

   ```bash
   ./simulation.sh <input fits file>
   ```
   This will create a directory named `simulated_visibility` which will contain the simulated visibilities for angular power spectrum.

   If this shows permission error, run the following command in the terminal :

   ```bash
   chmod +x ./simulation.sh
   ```
   If this shows an error that python command is not found then change `python` in `simulation.sh` to `python3`.
3. Now set `blt.Amp` and `blt.beta` in `params_sim.py` file to 1 and 0 respectively for the simulation for UAPS. Run the `simulation.sh` bash script again in the terminal with input fits file :

   ```bash
   ./simulation.sh <input fits file>
   ```
   This will create simulations for UAPS in the same `./simulated_visibility` directory.
4. You can change `blt.Nrea` in the `params_sim.py` file to your required no of realizations.
5. Run the `GV.sh` bash script in the terminal :

   ```bash
   ./GV.sh
   ```
   This will create `GV-data` directory containing the gridded visibilities for APS.

   If this shows permission error, run the following command in the terminal :

   ```bash
   chmod +x ./GV.sh
   ```
6. Now open the notebook file `TGE_GVtoPS.ipynb` for furthur analysis of the simulated data.
