"""WRF registry.ssib history names/flags, independent of model values.

Snow-layer variables carry restart flag r only. Standard soil/surface fields
retain their existing WRF metadata; SIB* names are the quoted NetCDF names.
"""
REGISTRY_SHA256 = '7fa144a8741211edf39adf8363db4f8f9022861a4c758644f53867d100ca02f6'
SSIB_EXTRA_HISTORY = {'SIBBR': 'rib', 'SIBEGS': 'xegs', 'SIBECI': 'xeci', 'SIBECT': 'xect', 'SIBEGI': 'xegi', 'SIBEGT': 'xegt', 'SIBSUP': 'xsup', 'SIBLUP': 'xlup', 'SIBWAT': 'xwat', 'SIBSHC': 'xhcx', 'SIBSHG': 'xhgx', 'SIBLAI': 'xzlt', 'SIBVCF': 'xvcf', 'SIBZ00': 'xxz0', 'SIBVEG': 'xveg'}
SSIB_HISTORY_METADATA = {'SIBBR': ('SSiB Bulk Richardson Number', '', 'Z'), 'SIBEGS': ('SSiB evaporation from soil', '', 'Z'), 'SIBECI': ('SSiB evaporation from interception', '', 'Z'), 'SIBECT': ('SSiB evaporation from transpiration', '', 'Z'), 'SIBEGI': ('SSiB evaporation from xxx', '', 'Z'), 'SIBEGT': ('SSiB evaporation from snow', '', 'Z'), 'SIBSUP': ('SSiB short wave rad. up ', '', 'Z'), 'SIBLUP': ('SSiB long wave rad. up ', '', 'Z'), 'SIBWAT': ('SSiB total soil moisture content', '', 'Z'), 'SIBSHC': ('SSiB sensible heat from canopy', '', 'Z'), 'SIBSHG': ('SSiB sensible heat from ground', '', 'Z'), 'SIBLAI': ('SSiB leaf area index', '', 'Z'), 'SIBVCF': ('SSiB vegetation cover', '', 'Z'), 'SIBZ00': ('SSiB surface roughness length', '', 'Z'), 'SIBVEG': ('SSiB vegetation map', '', 'Z')}
SSIB_STANDARD_HISTORY = {'SNOW': 'snoa', 'CANWAT': 'snob', 'SFROFF': 'roff', 'ALBEDO': 'bedo', 'SNOWH': 'snowdepth', 'GRDFLX': 'ghtflx'}
