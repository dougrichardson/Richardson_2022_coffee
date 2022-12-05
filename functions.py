import xarray as xr
import numpy as np
import pandas as pd
import geopandas

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import cartopy
import cartopy.crs as ccrs

import string

# ============================================================================
# Masks and tools
# ============================================================================
def get_combined_mask(dataset):
    """
    Gets the coffee growing region mask on the desired grid ('gpcc', 'berkeley' or 'era5').
    """
    
    growing_calendar = pd.read_csv('/g/data/xv83/dr6273/work/projects/coffee/data/coffee_country_growing_calendar_extended.csv',
                               index_col=0)
    
    if dataset == 'berkeley':
        dataset = 'gpcc' # berkely data uses GPCC mask
    
    coffee_mask = xr.open_dataset('/g/data/xv83/dr6273/work/projects/coffee/data/'+dataset+'_coffee_mask.nc')
    coffee_mask = coffee_mask.sel(abbrevs=np.unique(growing_calendar.abbrevs))
    return  coffee_mask.arabica + coffee_mask.robusta
    
def get_grid_area(dataset):
    """
    Calculates the area of coffee growing regions on the desired grid ('gpcc', 'berkeley' or 'era5')
    """
    
    if dataset == 'berkeley':
        dataset = 'gpcc' # berkely data uses GPCC mask
    # Area of each grid cell calculated using CDO
    grid_area = xr.open_dataset('/scratch/xv83/dr6273/data/'+dataset+'_grid_area.nc')
    grid_area = grid_area.cell_area
    if dataset == 'era5':
        return grid_area.rename({'latitude': 'lat', 'longitude': 'lon'})
    else:
        return grid_area
    
def get_n_Brazil_boundary():
    """
    Gets the boundary of the North Brazil shapefile
    """
    
    gdf = geopandas.read_file('/g/data/xv83/dr6273/work/projects/coffee/data/brazil_shapefiles/n_brazil.shp')
    gdf['new_column'] = 0
    return gdf.dissolve(by='new_column')

def get_se_Brazil_boundary():
    """ Gets the boundary of the Southeast Brazil shapefile """
    
    gdf = geopandas.read_file('/g/data/xv83/dr6273/work/projects/coffee/data/brazil_shapefiles/se_brazil.shp')
    gdf['new_column'] = 0
    return gdf.dissolve(by='new_column')
    
    
def get_country_order():
    """
    Order of coffee growing regions used in study
    """
    
    return {'BRS_0': 'Brazil S',
            'CO_1': 'Colombia 1',
            'CO_2': 'Colombia 2',
            'ET_3': 'Ethiopia',
            'HN_5': 'Honduras',
            'PE_8': 'Peru',
            'GT_4': 'Guatemala',
            'MX_6': 'Mexico',
            'NI_7': 'Nicaragua',   
            'VN_14': 'Vietnam',
            'BRN_9': 'Brazil N',
            'INDO_11': 'Indonesia',
            'UG_12': 'Uganda 1',
            'UG_13': 'Uganda 2',
            'IND_10': 'India'}

# ============================================================================
# Data wrangling
# ============================================================================

def open_era_data(root_path, variable, years, concat_dim='time'):
    """
    Open ERA5 data from NCI
    """
    ds_list = []
    for year in years:
        fp = root_path+variable+'/'+str(year)+'/*.nc'
        ds_list.append(xr.open_mfdataset(fp))
    return xr.concat(ds_list, dim=concat_dim)

def detrend_dim(da, dim, deg=1):
    """
    Detrend along a single dimension.
    
    Author: Ryan Abernathy
    From: https://gist.github.com/rabernat/1ea82bb067c3273a6166d1b1f77d490f
    """
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

def detrend(da, dims, deg=1):
    """
    Detrend along a multiple dimensions.
    Only valid for linear detrending (deg=1)
    
    Author: Ryan Abernathy
    From: https://gist.github.com/rabernat/1ea82bb067c3273a6166d1b1f77d490f
    """
    da_detrended = da
    for dim in dims:
        da_detrended = detrend_dim(da_detrended, dim, deg=deg)
    return da_detrended

def detrend_wrapper(da, dims=['time']):
    """
    Apply detrend, maintain attributes and chunk
    """
    
    dtype = da.dtype
    with xr.set_options(keep_attrs=True):
        da_dt = detrend(da, dims=dims)
        da_dt = da_dt.astype(dtype)
        da_dt = da_dt.assign_attrs({'short_name': da.attrs['short_name']+' (detrended)',
                                    'long_name': da.attrs['long_name']+' (detrended)'})
        return da_dt.chunk(da.chunks)

def sel_coffee_subregion(ds, dataset, species='both', lat_name='lat', lon_name='lon'):
    """
    Selects a region encompassing the coffee growing regions.
    
    Can select the coffee species ('arabica', 'robusta' or 'both') and the desired grid ('gpcc', 'berkeley' or 'era5').
    """
    
    def sum_and_bool(da):
        return da.sum('abbrevs').astype('bool')
    
    if dataset == 'berkeley':
        dataset = 'gpcc' # berkely data uses GPCC mask
    
    coffee_mask = xr.open_dataset('/g/data/xv83/dr6273/work/projects/coffee/data/'+dataset+'_coffee_mask.nc')
    
    if species == 'both':
        mask = sum_and_bool(coffee_mask.arabica + coffee_mask.robusta)
    elif species == 'arabica':
        mask = sum_and_bool(coffee_mask.arabica)
    elif species == 'robusta':
        mask = sum_and_bool(coffee_mask.robusta)
    else:
        raise ValueError("Incorrect species. Should be 'arabica', 'robusta', or 'both'.")
    
    lats = mask.lat.values
    lons = mask.lon.values
    
    first_lat = lats[np.where(mask.sum('lon') > 0)[0][0]]
    last_lat = lats[-np.where(mask.sum('lon').sortby('lat', ascending=True) > 0)[0][0]]
    
    first_lon = lons[np.where(mask.sum('lat') > 0)[0][0]]
    last_lon = lons[-np.where(mask.sum('lat').sortby('lon', ascending=False) > 0)[0][0]]
    
    return ds.sel({lat_name: slice(first_lat, last_lat), lon_name: slice(first_lon, last_lon)})

# ============================================================================
# Aggregating data over required periods
# ============================================================================

def get_seasonal_climate_data(da, phase, coffee_df):
    """
    Get climate data over months of interest. This will sum for phase='Annual', as in the
    study we look at 12-monthly precipitation totals. For other phases, a mean is computed.
    
    coffee_df should be a pandas DataFrame with columns 'abbrevs',
    'Start', 'Finish' and 'Phase'.
    """
    
    if phase == 'Annual':
        phase_df = coffee_df.loc[(coffee_df['Phase'] == 'Growing')].copy()
        other_phase_df = coffee_df.loc[(coffee_df['Phase'] != 'Flowering')].copy()
    else:
        phase_df = coffee_df.loc[(coffee_df['Phase'] == phase)].copy()
        other_phase_df = coffee_df.loc[(coffee_df['Phase'] != phase)].copy()
    n_seasons = len(phase_df)
    da_list = []
    
    for i in range(len(phase_df)):

        abbrev = phase_df.iloc[i]['abbrevs']

        phase_start = phase_df.iloc[i]['Start']
        phase_end = phase_df.iloc[i]['Finish']
        phase_length = int((phase_end - phase_start) % 12 + 1)
        
        # If phase is 'Annual', we want the annual totals (as only rainfall is required for annual in this analysis).
        #  For 'Growing' or 'Flowering', we want the seasonal mean.
        # Note: we could have neater code here using da.rolling().reduce(function), but
        # using in-built .rolling().mean() and .rolling().sum() is faster.
        if phase == 'Annual':
            da_roll = da.rolling(time=12).sum()
        else:
            da_roll = da.rolling(time=phase_length, min_periods=1).mean()
        da_roll = da_roll.isel(time=da_roll.groupby('time.month').groups[phase_end])

        # Set time index to align with yields (Flowering precedes Growing)
        # For growing, this is just the same year, so set index to beginning of year
        # If Finish Growing > Finish Flowering, flowering occurs in the same year as harvest.
        # If Finish Growing < Finish Flowering, then the flowering season occurrs in the calendar year prior to harvest,
        #  so set the year label to be the same as the harvest i.e. the following year.
        if (phase == 'Growing') | (phase == 'Annual'):
            da_roll['time'] = [pd.to_datetime(d) - pd.tseries.offsets.YearBegin() for d in da_roll['time'].values]
        else:
            yield_this_year = other_phase_df.iloc[i]['Finish'] > phase_end # the first term selects the right row from Growing

            if yield_this_year:
                da_roll['time'] = [pd.to_datetime(d) - pd.tseries.offsets.YearBegin() for d in da_roll['time'].values]
            else:
                da_roll['time'] = [pd.to_datetime(d) + pd.tseries.offsets.YearBegin() for d in da_roll['time'].values]
        
        da_roll = da_roll.assign_attrs(da.attrs)
        da_roll = da_roll.expand_dims({'season_id': [abbrev+'_'+str(i)]})
        
        da_list.append(da_roll)
        
    return xr.concat(da_list, dim='season_id')

def write_seasonal_climate_data(da,
                                output_var_name,
                                output_fp,
                                coffee_phase,
                                coffee_df,
                                event_type,
                                event_thresh):
    """
    Calculates climate hazard events for each year.

    event_type describes how events are defined. Should be 'upper_tail', 'lower_tail',
    or 'both_tails'.

    event_thresh is the threshold used to define events for event_type='upper_tail',
    event_type='lower_tail', or event_type='both_tails'. If standard deviations from
    the mean should be used, this should be a string in the format 'x_std'. For absolute
    values, an value is required for 'upper_tail' and 'lower_tail', and a list of two
    elements is required for 'both_tails', with the lower threshold in the first position.

    Note that this ignores the fact that Flowering seasons for some regions are undefined,
    as they include data prior to 1979. But to keep the arrays small, we use boolean
    arrays.
    """

    da_seasonal = get_seasonal_climate_data(da=da, phase=coffee_phase, coffee_df=coffee_df)
    da_seasonal = da_seasonal.assign_attrs(da.attrs)
    da_seasonal = da_seasonal.chunk({'time': -1, 'season_id': 8})
        
    # Get events
    if event_type == 'upper_tail':
        if isinstance(event_thresh, str):
            n_stds = float(event_thresh.split('_')[0])
            da_events_thresh = da_seasonal.mean('time') + n_stds * da_seasonal.std('time')
        else:
            da_events_thresh = event_thresh
                        
        # This is where we might mislabel a NaN as False - see docstring.
        da_events = xr.where(da_seasonal > da_events_thresh, True, False)
        
    elif event_type == 'lower_tail':
        if isinstance(event_thresh, str):
            n_stds = float(event_thresh.split('_')[0])
            da_events_thresh = da_seasonal.mean('time') - n_stds * da_seasonal.std('time')
        else:
            da_events_thresh = event_thresh
            
        da_events = xr.where(da_seasonal < da_events_thresh, True, False)
        
    elif event_type == 'both_tails':
        if isinstance(event_thresh, str):
            n_stds = float(event_thresh.split('_')[0])
            da_events_thresh_lower = da_seasonal.mean('time') - n_stds * da_seasonal.std('time')
            da_events_thresh_upper = da_seasonal.mean('time') + n_stds * da_seasonal.std('time')
        else:
            da_events_thresh_lower = event_thresh[0]
            da_events_thresh_upper = event_thresh[1]
            
        # Rather than boolean, we need this to be integers to allow for upper- and lower-tail events.
        da_events = xr.where(da_seasonal > da_events_thresh_upper, 1, 0)
        da_events = xr.where(da_seasonal < da_events_thresh_lower, -1, da_events)
        
    else:
        raise ValueError("Incorrect event_tail. Should be 'upper_tail', 'lower_tail' or 'both_tails'.")
    
    ds = da_seasonal.to_dataset(name=output_var_name)
    
    ds = ds.assign({'_'.join(['event', str(event_thresh)]): da_events})

    ds.to_zarr(output_fp, mode='w', consolidated=True)
    
def process_and_write(ds,
                      dataset,
                      var,
                      event_list,
                      detrend,
                      coffee_df,
                      spatial_field=True,
                      chunks={'time': -1, 'lat': -1, 'lon': 220}):
    
    """
    Processes and writes to file the climate hazards computations.
    
    event_list should be an list of 3 lists, where each internal list represents the phase (str),
    event type (str) and event threshold (float or str).
    """
        
    event_phases = [x[0] for x in event_list]
    event_types = [x[1] for x in event_list]
    event_threshs = [x[2] for x in event_list]
    
    output_var_name = var
    
    da = ds[var]
    if spatial_field:
        da = sel_coffee_subregion(da, dataset)
        da = da.chunk(chunks)
    
    if detrend:
        da = detrend_wrapper(da)
        output_var_name += '_detrended'
        
    for event_phase, event_type, event_thresh in zip(event_phases, event_types, event_threshs):

        output_fn = '_'.join([dataset, output_var_name, event_phase, event_type, str(event_thresh)]) + '.zarr'
        output_fp = '/g/data/xv83/dr6273/work/projects/coffee/data/'

        write_seasonal_climate_data(da=da,
                                    output_var_name=output_var_name,
                                    output_fp=output_fp + output_fn,
                                    coffee_phase=event_phase,
                                    coffee_df=coffee_df,
                                    event_type=event_type,
                                    event_thresh=event_thresh)
        
# ============================================================================
# Event statistics
# ============================================================================

def calculate_event_statistics(da, dataset, absolute_threshold=False, threshold_value='std'):
    """
    Calculate the proportion and area of a subregion that experiences an event
    
    absolute_threshold is a boolean indicating whether or not to use an absolute
    value as the proportion of land area that triggers an event.
    
    threshold_value is the proportion of land area required to trigger
    a regional event. If absolute_threshold == True, threshold_value is added to
     the mean areal proportion experiencing an event each year. 
     Default is 'std', corresponding to one standard deviation
    above the mean. Otherwise this should be a float between 0 and 1.
    If absolute_threshold == True, threshold_value is simply the proportion of area
    that triggers an event.
    """
    
    mask = get_combined_mask(dataset)
    grid_area = get_grid_area(dataset)
    coffee_df = pd.read_csv('/g/data/xv83/dr6273/work/projects/coffee/data/coffee_country_growing_calendar_extended.csv',
                               index_col=0)
    
    event_name = [s for s in list(da.data_vars) if "event_" in s][0]
    ds_list = []
    for i, abbrev in enumerate(coffee_df.loc[(coffee_df.Phase == 'Flowering'),
                                              'abbrevs']):
        s_id = abbrev+'_'+str(i)
        area_of_country = grid_area.where(mask.sel(abbrevs=abbrev) == True).sum()
        area_of_event = grid_area.where(da[event_name].sel(season_id=s_id).where(mask.sel(abbrevs=abbrev) == True) == True).sum(['lat', 'lon'])
        
        # Proportions of each country experiencing an event
        country_proportion_of_event = area_of_event / area_of_country.values
        
        # Get threshold of anomlously widespread events
        if absolute_threshold:
            proportion_thresh = threshold_value
        else:
            if threshold_value == 'std':
                proportion_thresh = country_proportion_of_event.mean('time') + country_proportion_of_event.std('time')
            else:
                proportion_thresh = country_proportion_of_event.mean('time') + threshold_value
        country_event = xr.where(country_proportion_of_event > proportion_thresh, True, False)
        
        # Combine into DataSet
        area_of_event = area_of_event.to_dataset(name='area_of_event')
        country_proportion_of_event = country_proportion_of_event.to_dataset(name='country_proportion_of_event')
        country_event = country_event.to_dataset(name='event')
        
        ds = area_of_event.merge(country_proportion_of_event).merge(country_event)
    
        ds_list.append(ds)
        
    return xr.concat(ds_list, dim='season_id')

def calculate_event_proportion(event_area_da, region_area_name, dataset):
    """
    Calculates the proportion of area experiencing an event.
    As Colombia and Uganda have two separate seasons each, their area is
    counted twice.
    """
    
    mask = get_combined_mask(dataset)
    grid_area = get_grid_area(dataset)
    
    if region_area_name == 'global':
        region_area = grid_area.where(mask.sum('abbrevs') == True).sum().values
        region_area += grid_area.where(mask.sel(abbrevs='CO') == True).sum().values \
                        + grid_area.where(mask.sel(abbrevs='UG') == True).sum().values
    elif region_area_name == 'arabica':
        region_area = grid_area.where(mask.arabica.sum('abbrevs') == True).sum().values
        region_area += grid_area.where(mask.sel(abbrevs='CO') == True).sum().values
    elif region_area_name == 'robusta':
        region_area = grid_area.where(mask.robusta.sum('abbrevs') == True).sum().values
        region_area += grid_area.where(mask.sel(abbrevs='UG') == True).sum().values
    else:
        raise ValueError("region_area_name should be 'global', 'arabica', or 'robusta'.")
        
    return event_area_da / region_area

def avg_risk(da, risk_name, years=slice('1980', '2021'), time_name='time'):
    return da.sel({time_name: years}).country_proportion_of_event.mean(time_name).expand_dims({'risk': [risk_name]})

def combine_n_events(risks_dict_list):
    """
    Concatenate number of events for different risks
    """
    
    def get_n_events(d):
        keys = list(d.keys())
        da = xr.full_like(d[keys[0]].event.astype('int8'), 0)
        for r1_key, r1_value in zip(keys, d.values()):
            da += r1_value.event.astype('int8')
        return da
    
    n_events = []
    for risks_dict in risks_dict_list:
        n_events.append(get_n_events(risks_dict))
    return xr.concat(n_events, dim='season_id').astype('int16')


# ============================================================================
# Bootstrapping
# ============================================================================

def estimate_L(da):
    """
    Estimates block length L for each grid box of da.
    """
    from statsmodels.tsa.stattools import acf
    
    def acf_lag1(x):
        if np.sum(~np.isnan(x)) == 0: # if all NaNs
            return np.nan
        else:
            x = x[~np.isnan(x)]
            return acf(x, nlags=1)[-1]
    
    n = len(da.time.values)
    
    # DataArray of lag1 ACF coefficients
    rho_da = xr.apply_ufunc(acf_lag1, da, input_core_dims=[['time']], output_core_dims=[[]], vectorize=True, dask='forbidden')
    
    # DataArray of effective sample size
    n_eff_da = n * ((1 - rho_da) / (1 + rho_da))
    
    # Initialise guess for block length
    Ls_da = xr.full_like(rho_da, 1)
    for i in range(10): # iterate to get estimate of L
        L_da = (n - Ls_da + 1) ** ( (2/3) * (1 - n_eff_da / n) )
        Ls_da = L_da
    
    return np.ceil(L_da) # round up to get block length
    
def get_quantile(obs, bootstrap):
    """
    Returns the quantile of obs in bootstrap
    """
    if np.isnan(obs):
        return np.nan
    else:
        return np.searchsorted(np.sort(bootstrap), obs) / len(bootstrap)

# ============================================================================
# Plotting
# ============================================================================

letters = list(string.ascii_lowercase)

def get_plot_params():
    """
    Get the plotting parameters used for figures
    """
    FONT_SIZE = 7
    COASTLINES_LW = 0.5
    LINEWIDTH = 1.3
    PATHEFFECT_LW_ADD = LINEWIDTH * 1.8

    return {'lines.linewidth': LINEWIDTH,
            'hatch.linewidth': 0.5,
            'font.size': FONT_SIZE,
            'legend.fontsize' : FONT_SIZE - 1,
            'legend.columnspacing': 0.7,
            'legend.labelspacing' : 0.03,
            'legend.handlelength' : 1.,
            'axes.linewidth': 0.5}
    
def var_summary(da, event_name, sum_or_mean, season_ids, dataset, years):
    """
    Compute summary statistics of grid cell-level events.
    """
    mask = get_combined_mask(dataset)
    abbrevs = [i.split('_')[0] for i in season_ids]
    da_list = []
    for k, a in zip(season_ids, abbrevs):
        events = da.sel(time=years, season_id=k)[event_name]
        if sum_or_mean == 'sum':
            events = events.where(mask.sel(abbrevs=a)).sum('time')
        elif sum_or_mean == 'mean':
            events = events.where(mask.sel(abbrevs=a)).mean('time')
        else:
            raise ValueError("sum_or_mean should be 'sum' or 'mean'.")
        da_list.append(events)
    events_da = xr.concat(da_list, dim='season_id')

    return events_da.sum('season_id').where(mask.sel(abbrevs=abbrevs).sum('abbrevs'))

def risks_map(events_or_distance, var_dict, event_name, season_ids, dataset,
              save_fig, filename, years=slice('1980','2020')):
    """
    Plot summary statistics for grid cell-level hazard events.
    """
    plt_params = get_plot_params()
    n_brazil = get_n_Brazil_boundary()
    mask = get_combined_mask(dataset)
    
    if events_or_distance == 'events':
        aggregation_method = 'sum'
        nYears = len(range(int(years.start), int(years.stop) + 1))
        vmax = nYears
        cbar_ticks = range(0, 41, 5)
        cbar_extend = 'neither'
        cbar_label = 'Frequency [years]'
    elif events_or_distance == 'distance':
        aggregation_method = 'mean'
        vmax = 2
        cbar_ticks = np.arange(0, 2.1, 0.25)
        cbar_extend = 'max'
        cbar_label = r'Distance surpassing threshold [$\sigma$]'
    else:
        raise ValueError("events_or_distance must be 'events' or 'distance'.")
    
    fig, ax = plt.subplots(3, 2, figsize=(6.9, 4), subplot_kw={'projection': ccrs.PlateCarree()})
    
    for i, (var_key, var_da) in enumerate(zip(var_dict.keys(), var_dict.values())):
        var_da = var_da.sel(time=years)
        
        plot_da = var_summary(var_da, 'event', aggregation_method, var_da.season_id.values, 'gpcc', years)
        
        ax.flatten()[i].set_extent((-117, 142, 36, -35), crs=ccrs.PlateCarree())
        ax.flatten()[i].coastlines(lw=plt_params['lines.linewidth']/3)
        n_brazil.boundary.plot(ax=ax.flatten()[i], color='r', lw=plt_params['lines.linewidth']/4)
        ax.flatten()[i].add_feature(cartopy.feature.BORDERS, lw=plt_params['lines.linewidth']/4)
        
        ax.flatten()[i].plot((26, 26), (8, -90), color='r', ls='-', lw=plt_params['lines.linewidth']/1.2)
        ax.flatten()[i].plot((48, 48), (90, 0), color='r', ls='-', lw=plt_params['lines.linewidth']/1.2)
        ax.flatten()[i].plot((26.5, 48), (8, 0), color='r', ls='-', lw=plt_params['lines.linewidth']/1.2)
        
        p = plot_da.plot(ax=ax.flatten()[i], vmin=0, vmax=vmax, add_colorbar=False,
                        rasterized=True)
        
        left_title = var_key.split('__')[0]
        right_title = var_key.split('__')[-1]
        ax.flatten()[i].text(0.25, 1.1, left_title, ha='center', transform=ax.flatten()[i].transAxes)
        ax.flatten()[i].text(0.8, 1.1, right_title, ha='center', transform=ax.flatten()[i].transAxes)
        
        if i == 0:
            ax.flatten()[i].text(0.3, 0.05, 'Arabica', transform=ax.flatten()[i].transAxes)
            ax.flatten()[i].text(0.65, 0.05, 'Robusta', transform=ax.flatten()[i].transAxes)
        
        ax.flatten()[i].text(0.02, 0.05, letters[i], weight='bold',
                             transform=ax.flatten()[i].transAxes)
        
    cb_ax1 = fig.add_axes([0.23, 0.11, 0.6, 0.03])
    cb1 = fig.colorbar(p, cax=cb_ax1, orientation='horizontal',
                       ticks=cbar_ticks, extend=cbar_extend)
    cb1.ax.set_xlabel(cbar_label)
    
    plt.subplots_adjust(hspace=0.01, wspace=0.01)
    
    if save_fig:
        plt.savefig('./figures/'+filename, format='pdf', dpi=400, bbox_inches='tight')
        
def plot_combined_phase_extremes(plot_dict_list, y_order, save_fig, filename,
                                years=slice('1980', '2020')):
    """
    Plot time series of hazard events
    """
    plt_params = get_plot_params()
    cmap = matplotlib.cm.get_cmap('viridis')
    cmapBig = matplotlib.cm.get_cmap('viridis', 512)
    
    figsize = (6.9, 7.5)
    
    with plt.rc_context(plt_params):        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(nrows=3, ncols=1, figure=fig,height_ratios=[1,1,1])

        
        for i, plot_dict in enumerate(plot_dict_list):
            f_ds = plot_dict['da1'].sel(time=years)
            g_ds = plot_dict['da2'].sel(time=years)
            event_categories_1 = plot_dict['event_categories_1']
            event_categories_2 = plot_dict['event_categories_2']
            cmap_max = plot_dict['cmap_max']
            title = plot_dict['title']
    
            time = f_ds.time
            cbar_max = len(event_categories_1) + 0.5
            
            newcmap = matplotlib.colors.ListedColormap(cmapBig(np.linspace(0, 1, cmap_max//2)))
            norm = matplotlib.colors.BoundaryNorm(np.arange(cbar_max + 0.5), newcmap.N)
        
            plot_data = f_ds.event + g_ds.event * 2
            plot_data = plot_data.sel(season_id=list(y_order.keys())) # reorder
            
            ax = fig.add_subplot(gs[i])

            lego_plot = ax.pcolormesh(plot_data, cmap=cmap, norm=norm)
            lego_plot2 = ax.pcolormesh(plot_data, cmap=cmap, norm=norm) # bug means we need to plot twice to have two colorbars

            ax.invert_yaxis()
            ax.set_yticks(np.arange(0.5, len(f_ds.season_id)))
            ax.set_yticklabels(y_order.values())

            ax.set_xlim(0, len(time))
            ax.set_xticks(np.arange(0.5, len(time)))
            if i == 2:
                xticklabels = []
                for t in time.dt.year.values[::2]:
                    xticklabels.append(t)
                    xticklabels.append('')
                ax.set_xticklabels(xticklabels[:-1], rotation=45)
            else:
                ax.set_xticklabels([])

            ax.set_title(title, loc='left')

            ax.text(-0.135, 0.7, 'Arabica', transform=ax.transAxes, rotation=90, ha='center', va='center')
            ax.text(-0.135, 0.2, 'Robusta', transform=ax.transAxes, rotation=90, ha='center', va='center')

            ax.annotate('', xy=(-0.15, 0.4), xycoords='axes fraction', xytext=(1.22, 0.4), 
                arrowprops=dict(arrowstyle="-", ls=':', lw=plt_params['lines.linewidth']-0.5), zorder=0)
            ax.annotate('', xy=(-0.005, 0.4), xycoords='axes fraction', xytext=(1.005, 0.4), 
                arrowprops=dict(arrowstyle="-", color='white', ls=':', lw=plt_params['lines.linewidth']-0.5))
            ax.annotate('', xy=(-0.125, 0.), xycoords='axes fraction', xytext=(-0.125, 1), 
                arrowprops=dict(arrowstyle="-", ls=':', lw=plt_params['lines.linewidth']-0.5))

            axins1 = inset_axes(ax, width="5%", height="95%",
                                bbox_to_anchor=(.54, .5, .5, .4),
                                bbox_transform=ax.transAxes, loc='lower right', borderpad=0)
            cb1 = fig.colorbar(lego_plot, cax=axins1, orientation='vertical', ticks=np.arange(0.5, cbar_max, 1))
            cb1.ax.set_yticklabels(event_categories_1)

            axins2 = inset_axes(ax, width="5%", height="95%",
                                bbox_to_anchor=(.54, -0.01, .5, .4),
                                bbox_transform=ax.transAxes, loc='lower right', borderpad=0)
            cb2 = fig.colorbar(lego_plot2, cax=axins2, orientation='vertical', ticks=np.arange(0.5, cbar_max, 1))
            cb2.ax.set_yticklabels(event_categories_2)

        plt.subplots_adjust(hspace=0.18)
            
        if save_fig:
            plt.savefig('./figures/'+filename, format='pdf', dpi=400, bbox_inches='tight')
            
def plot_n_signed_events(risks_dict_list, signed_risks_dict_list, y_order, save_fig, filename):
    """
    Plot time series of combined hazards
    """
    plt_params = get_plot_params()
    
    n_events = combine_n_events(risks_dict_list)
    signed_n_events = combine_n_events(signed_risks_dict_list)
    equal_events = xr.where((n_events > 0) & (signed_n_events == 0), n_events, np.nan) # save array of equal warm/dry and cold/wet events
    print(equal_events.min().values, equal_events.max().values)
    n_events = xr.where(signed_n_events < 0, n_events * -1, n_events)
    # Note that this means years with equal warm/dry and cold/wet will be signed as the former.
    #. We draw over these years with equal_events.
    n_events = n_events.sel(season_id=list(y_order.keys())) # reorder
    
    n_min = n_events.min().values
    n_max = n_events.max().values
    n_abs_max = np.max(np.abs([n_min, n_max]))
    print(n_min, n_max)
        
    figsize = (6.9, 3)
    time = n_events.time

    with plt.rc_context(plt_params):
        cmap = plt.cm.BrBG_r
        norm = matplotlib.colors.BoundaryNorm(np.arange(-n_abs_max, n_abs_max+2), cmap.N)

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(ncols=2, nrows=2, height_ratios=[.2,1], width_ratios=[1,.1])
        
        ###=========================================== Lego plot
        ax = fig.add_subplot(gs[1,0])
        ax.set_zorder(2)

        lego_plot = ax.pcolormesh(n_events, cmap=cmap, norm=norm)
        equal_plot = ax.pcolormesh(equal_events, cmap='spring', vmin=0, vmax=4)

        ax.invert_yaxis()
        ax.set_yticks(np.arange(0.5, len(n_events.season_id)))
        ax.set_yticklabels(list(y_order.values()))
#         ax.set_ylabel('Coffee seasons\n\n')

        ax.set_xticks(np.arange(0.5, len(time)))
        xticklabels = []
        for t in time.dt.year.values[::2]:
            xticklabels.append(t)
            xticklabels.append('')
        ax.set_xticklabels(xticklabels[:-1], rotation=45)
        
        ax.text(-0.152, 0.7, 'Arabica', transform=ax.transAxes, rotation=90, ha='center', va='center')
        ax.text(-0.152, 0.2, 'Robusta', transform=ax.transAxes, rotation=90, ha='center', va='center')
        ax.annotate('', xy=(-0.165, 0.4), xycoords='axes fraction', xytext=(1.13, 0.4), 
            arrowprops=dict(arrowstyle="-", ls=':', lw=plt_params['lines.linewidth']/1.5))
        ax.annotate('', xy=(-0.14, 0.), xycoords='axes fraction', xytext=(-0.14, 1), 
            arrowprops=dict(arrowstyle="-", ls=':', lw=plt_params['lines.linewidth']/1.5))
        
        cb_ax1 = fig.add_axes([0.93, 0.22, 0.02, 0.42])
        cb1 = fig.colorbar(lego_plot, cax=cb_ax1, orientation='vertical', ticks=np.arange(-n_abs_max+0.5, n_abs_max+1.5, 1))
        cb1.ax.set_yticklabels(np.abs(np.arange(-n_abs_max, n_abs_max+1, 1)))
        cb1.ax.set_xlabel('Number\nof events')
        fig.text(1.0, 0.23, 'Majority\ncold or wet', rotation=270, ha='center')
        fig.text(1., 0.45, 'Majority\nwarm or dry', rotation=270, ha='center')
        
        ax.plot((1.189, 1.197), (0.95, 0.95), c='k', lw=0.75,
                clip_on=False, transform=ax.transAxes)
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], marker='s', ms=10, color=plt.cm.spring(0.5),
                               markeredgecolor='k', markeredgewidth=0.5)]
        ax.legend(custom_lines, [''], frameon=False, loc=[1.155, 0.9],
                 fontsize=plt_params['font.size'])
        ax.text(1.205, 0.955, 'One of\neach', rotation=0, ha='left', va='center',
                transform=ax.transAxes)
        
        
        ###=========================================== Top bar and line plot
        ax = fig.add_subplot(gs[0,0])
        
        # Bar plot
        c_ = [0.25, 0.75] # where in cmap to get colors from
        
        negatives = xr.where(n_events < 0, np.abs(n_events), 0)
        positives = xr.where(n_events > 0, n_events, 0)
        
        year_counts_neg = negatives.sum('season_id')
        year_counts_pos = positives.sum('season_id')
        event_max = (year_counts_pos + year_counts_neg).max().values
        print(year_counts_neg.max().values)
        print(year_counts_pos.max().values)
                        
        ax.bar(time.dt.year.values, year_counts_neg, color=cmap(c_[0]), width=.8)
        ax.bar(time.dt.year.values, year_counts_pos, bottom=year_counts_neg, color=cmap(c_[1]), width=.8)
        
        ax.set_ylim(0, event_max)
        ax.set_yticks([0, event_max])
        ax.set_ylabel('Number of\nevents')
        
        ax.set_xticks([])
        ax.set_xlim(1979.5, 2020.5)
        
        for pos in ['top', 'right']:
            ax.spines[pos].set_visible(False)
            
        ###=========================================== Right bar plot
        ax = fig.add_subplot(gs[1,1])
        ax.set_zorder(1)
        
        country_counts_neg = negatives.sum('time')
        country_counts_pos = positives.sum('time')
        year_max = np.max(country_counts_neg + country_counts_pos)
                        
        ax.barh(n_events.season_id.values, country_counts_neg, color=cmap(c_[0]), height=.77)
        ax.barh(n_events.season_id.values, country_counts_pos, left=country_counts_neg, color=cmap(c_[1]), height=.77)
            
        ax.set_yticks([])
        ax.set_ylim(-0.5,14.5)
        ax.invert_yaxis()
        
        ax.set_xticks([0, year_max])
        ax.xaxis.tick_top()
        ax.set_xlabel('Number of\nyears')
        ax.xaxis.set_label_position('top') 
        
        for pos in ['bottom', 'right']:
            ax.spines[pos].set_visible(False)
            
        plt.subplots_adjust(hspace=0.12, wspace=0.03)
        
        if save_fig:
            plt.savefig('./figures/'+filename, format='pdf', dpi=400, bbox_inches='tight')
            
        return [year_counts_neg, year_counts_pos]