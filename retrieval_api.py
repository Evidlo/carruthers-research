from glide.common_components.view_geometry import SpaceCraft, CameraWFI, CameraNFI

cam = CameraWFI()
sc = SpaceCraft('2025-06-01', cam)

sc_list = gen_mission(num_obs=50, start='2025-06-01', duration='90', cam=[cam])

## Thick code
cam = CameraNFI()
sc_list = gen_mission(num_obs=48, start='2025-06-01', duration='1', cam=[cam])
sc_list.get_tp_lat_lon()


vg = [SpaceCraft, SpaceCraft, ...]
images = np.random.random((len(vg), 512, 512))

vg = [[SpaceCraft, SpaceCraft, ...], [SpaceCraft, SpaceCraft, ...]]
images = [np.random.random((50, 512, 512)), np.random.random((75, 1024, 1024))]

def retrieval(vg, images, thin_grid):
    """Do retrieval algorithm
    
    Args:
        vg (list[list[SpaceCraft]]): list of measurement viewing geometries for each sensor
        images (list[ndarray]): measurements for each sensor
        thin_grid (volume geometry?): 3D (r, e, a) GSE volume grid
        r_thick_grid
        
    Return:
        thot (ndarray)
        ...
    """

    # is therm_dens_2d actually a 2D array?
    # is grid always reported at exobase?
    
    # ----- Pratik -----
 
    science_nfi = science_binning(images)
    #solar_fluxes - ndarray, [24], solar_flux for each wedge solution
    #albedo_2ds - ndarray [32x32x24], [r x sza x wedge] source function -> average albedo over all wedges & time
    #therm_dens_2d - ndarray [32x32x24], [r x sza x wedge] H density -> L2 thermal H density
    #h_exos - ndarray [24], H density at exobase
    #exobase_grid - Grid, gse_lat [24], gse_lon [24], z_exo [24] GSE lat, lon, exobase altitude for each wedge
    #rsza_grid - Grid, r [32], sza[32] r & sza values for grid
    solar_fluxes, albedo_2ds, therm_dens_2ds, rsza_grid, h_exos, exobase_grid = thick_therm(vg, science_nfi)
    thick_grid = combine_grids(r_thick_grid, thin_grid) # take E and A from thin_grid
    therm_dens_3d_thick = 2d_to_3d(thick_grid, exobase_grid, h_exos) # exobase_grid has gse_lat gse_lon -- L2 data product, needs to be archived 
    # therm_dens_3d.shape --> (len(times), *thin_vol.shape)
    
    # ----- Evan ------
    
    # average all 2D retrieved albedos into single [32x32] profile, average albedos
    albedo_2d, solar_flux = average_albedos(albedo_2ds, solar_fluxes)
    albedo_3d = 2d_to_3d(thin_grid, rsza_grid, albedo_2d) # sweep 2D albedo to 3D
    tot_dens_3d_thin = thin(thin_grid, vg, images, albedo_3d, solar_flux) # L2 data product to be archived
    # tot_dens_3d.shape --> (len(times), *thin_vol.shape)
    
    
    tot_grid = combine_grids(thick_grid, thin_grid) # tot_grid is R, E, A.  check that E and A are same
    tot_dens_3d = functional_fit(rsza_gridrm_dens_3d_thick)
    tot_nontherm_dens_3d = tot_dens_3d - therm_dens_3d
    
    return tot_dens, tot_nontherm_dens_3d, therm_dens_3d, tot_grid, solar_fluxes
    