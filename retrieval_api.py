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

def retrieval(vg, images, times, thick_vol, thin_vol):
    """Do retrieval algorithm
    
    Args:
        vg (list[list[SpaceCraft]]): list of measurement viewing geometries for each sensor
        images (list[ndarray]): measurements for each sensor
        times (list[str]): retrieval times
        thick_vol (volume geometry?): 2D (az, sza, r) volume grid
        thin_vol (volume geometry?): 3D (X, Y, Z) GSE volume grid
        
    Return:
        thot (ndarray)
        ...
    """
  
    science_nfi = science_binning(images)
    solar_flux, albedo_2d, therm_dens_2d = thick_therm(thick_vol, vg, science_nfi)
    therm_dens_3d = 2d_to_3d(vg, therm_dens_2d) # vg has tp_lat tp_lon -- L2 data product, needs to be archived 
    # therm_dens_3d.shape --> (len(times), *thin_vol.shape)
    
    albedo_3d = 2d_to_3d(vg, albedo_2d) # vg has tp_lat tp_lon 
    tot_dens_3d = thin(thin_vol, vg, images, albedo_3d, solar_flux) # L2 data product to be archived
    # tot_dens_3d.shape --> (len(times), *thin_vol.shape)
    
    tot_dens_2d = 3d_to_2d_dens(vg, tot_dens_3d) # vg has tp_lat tp_lon of a given wedge
    # tot_dens_2d.shape --> (len(times), *thick_vol.shape)
    thot = thick_nontherm(thick_vol, vg, science_nfi, therm_dens_2d, tot_dens_2d) # L2 data product to be archived
    
    return thot
    
    