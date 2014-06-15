################################################################################################################################################
# IMPORT MODULES
################################################################################################################################################

#import general python tools
import argparse
import itertools
import operator
from operator import itemgetter
import sys, os, shutil
import os.path
import math

#import python extensions/packages to manipulate arrays
import numpy 				#to manipulate arrays
import scipy 				#mathematical tools and recipesimport MDAnalysis

#import graph building module
import matplotlib as mpl
mpl.use('Agg')
import pylab as plt
import matplotlib.cm as cm			#colours library
import matplotlib.colors as mcolors
import matplotlib.ticker
from matplotlib.ticker import MaxNLocator
from matplotlib.font_manager import FontProperties
fontP=FontProperties()

#import clustering algorithms
from sklearn.cluster import DBSCAN
import networkx as nx

#import MDAnalysis
import MDAnalysis
from MDAnalysis import *
import MDAnalysis.analysis
import MDAnalysis.analysis.leaflet
import MDAnalysis.analysis.distances

#set MDAnalysis to use periodic boundary conditions
MDAnalysis.core.flags['use_periodic_selections'] = True
MDAnalysis.core.flags['use_KDTree_routines'] = False

################################################################################################################################################
# RETRIEVE USER INPUTS
################################################################################################################################################

#create parser
#=============
version_nb="3.3.0"
parser = argparse.ArgumentParser(prog='cluster_lip', usage='', add_help=False, formatter_class=argparse.RawDescriptionHelpFormatter, description=\
'''
**********************************************
v''' + version_nb + '''
author: Jean Helie
git: https://github.com/jhelie/cluster_lip.git
**********************************************
	
[ Description ]

This script identifies lipid clusters in a gro file or throughout a trajectory
using either a connectivity based (networkX) or a density based (DBSCAN) algorithm
on the x,y,z coordinates of lipids headgroups.
	
It produces 3 types of outputs (only the 1st can be obtained without specifying groups, see note 5):
 - 2D plots: for each specie, time evolution of the cluster size each lipid is involved in
 - 1D plots: for each specie, time evolution of the % of lipids represented by each size group
 - stability statistics: for each specie, maximun nb of consecutive frames each size group existed for

[ Requirements ]

The following python modules are needed:
 - MDAnalysis
 - scikit learn (python-sklearn)

[ Notes ]

1. It's a good idea to trjconv the xtc first and only outputs the lipids of 
   interest, the script will run much faster. Also, use the -pbc mol option.	

2. Two clustering algorithms can be used:
   -connectivity: based on networkX, a lipid is considered in a cluster if its within a distance
                  less than --cutoff from another lipid. This means that a single lipid can potentially
                  act as a connectory between two otherwise disconnected lipid patches..
   -density: based on the DBSCAN algorithm implemented in scikit, a lipid is considered in a cluster
             if is surrounded by at least --neighbours other lipids within a radius of --radius.
             This density based approach is usually more suited to the detection of 'patches'. For 
	         subtleties on this algorithm (e.g. edge dectection) see its online documentation.

3. The most appropriate parameters for each algorithm depends on the lipid species of interest.
   For the density based algorithm the more compact the clusters, the smaller --radius and the higher
   --neighbours can be.

4. The code can easily be updated to add more lipids and forcefields. For now the following
   lipids can be dealt with:
    - Martini: DHPC,DHPE,DLPC,DLPE,DAPC,DUPC,DPPC,DPPE,DPPS,DPPG,DSPC,DSPE,POPC,POPE,POPS,POPG,PPCS,PIP2,PIP3,GM3
	
5. Clusters statistics can be binned by defining size groups (-g).
   Size groups should be specified in a file where each line has the format:
       lower_group_size,upper_group_size, colour
   and respect the following rules:
    - to specify an open ended group use 'max', e.g. '3,max,color'
    - groups should be ordered by increasing size and their boundaries should not overlap
    - boundaries are inclusive so you can specify one size groups with 'size,size,color'
    - colours MUST be specified, either as single letter code, hex code  or colormap name (see note 6)
    - in case a colormap is used its name must be specified as the color of each cluster
    - any cluster size not fallig within the specified size groups will be labeled as 'other' and
      coloured in grey (#C0C0C0).
	
6. If you don't want to use a custom colour map, standard matplotlib colormaps can be specified.
   Type cluster_lip --colour_maps to see a list of their names.

7. The size (or size group) of the cluster each lipid is detected to be involved can be visualised
   with VMD. This can be done either with pdb files (output frequency controled via -w flag) or with 
   the xtc trajectory.
     - pdb file: the clustering info for each protein is stored in the beta factor column. Just open
                 the pdb with VMD and choose Draw Style > Coloring Method > Beta 
     - xtc file: the clustering info is stored in a .txt file in /4_VMD/ and you can load it into the
                 user2 field in the xtc by sourcing the script 'set_user_fields.tcl' and running the 
                 procedure 'set_cluster_lip'

[ Usage ]

Option	      Default  	Description                    
-----------------------------------------------------
-f			: structure file [.gro]
-x			: trajectory file [.xtc]
-g			: cluster groups definition file, see note 5
-o			: name of output folder
-b			: beginning time (ns) (the bilayer must exist by then!)
-e			: ending time (ns)	
-t 		10	: process every t-frames
-w			: write annotated pdbs every [w] processed frames (optional, see note 7)
--smooth		: nb of points to use for data smoothing (optional)
--algorithm	density	: 'connectivity' or 'density' (see note 2)
--forcefield		: forcefield options, see note 4
--no-opt		: do not attempt to optimise leaflet identification (useful for huge system)

Algorithm options (see note 3)
-----------------------------------------------------
--cutoff 	8	: networkX cutoff distance for lipid-lipid contact (Angtrom)
--radius 	10	: DBSCAN search radius (Angtrom)
--neighbours 	5	: DBSCAN minimum number of neighbours within a circle of radius --radius	
 
Other options
-----------------------------------------------------
--colour_maps		: show list of standard colour maps (see note 6)
--version		: show version number and exit
-h, --help		: show this menu and exit
 
''')

#data options
parser.add_argument('-f', nargs=1, dest='grofilename', default=['no'], help=argparse.SUPPRESS)
parser.add_argument('-x', nargs=1, dest='xtcfilename', default=['no'], help=argparse.SUPPRESS)
parser.add_argument('-g', nargs=1, dest='cluster_groups_file', default=['no'], help=argparse.SUPPRESS)
parser.add_argument('-o', nargs=1, dest='output_folder', default=['no'], help=argparse.SUPPRESS)
parser.add_argument('-b', nargs=1, dest='t_start', default=[-1], type=int, help=argparse.SUPPRESS)
parser.add_argument('-e', nargs=1, dest='t_end', default=[10000000000000], type=int, help=argparse.SUPPRESS)
parser.add_argument('-t', nargs=1, dest='frames_dt', default=[10], type=int, help=argparse.SUPPRESS)
parser.add_argument('-w', nargs=1, dest='frames_write_dt', default=[1000000000000000], type=int, help=argparse.SUPPRESS)
parser.add_argument('--forcefield', dest='forcefield_opt', choices=['martini'], default='martini', help=argparse.SUPPRESS)
parser.add_argument('--algorithm', dest='m_algorithm', choices=['connectivity','density'], default='density', help=argparse.SUPPRESS)
parser.add_argument('--smooth', nargs=1, dest='nb_smoothing', default=[0], type=int, help=argparse.SUPPRESS)
parser.add_argument('--no-opt', dest='cutoff_leaflet', action='store_false', help=argparse.SUPPRESS)
#algorithm options
parser.add_argument('--cutoff', nargs=1, dest='cutoff_connect', default=[8], type=float, help=argparse.SUPPRESS)
parser.add_argument('--radius', nargs=1, dest='dbscan_dist', default=[17], type=float, help=argparse.SUPPRESS)
parser.add_argument('--neighbours', nargs=1, dest='dbscan_nb', default=[3], type=int, help=argparse.SUPPRESS)
#other options
parser.add_argument('--colour_maps', dest='show_colour_map', action='store_true', help=argparse.SUPPRESS)
parser.add_argument('--version', action='version', version='%(prog)s v' + version_nb, help=argparse.SUPPRESS)
parser.add_argument('-h','--help', action='help', help=argparse.SUPPRESS)

#store inputs
#============
args=parser.parse_args()
args.grofilename=args.grofilename[0]
args.xtcfilename=args.xtcfilename[0]
args.output_folder=args.output_folder[0]
args.frames_dt=args.frames_dt[0]
args.frames_write_dt=args.frames_write_dt[0]
args.t_start=args.t_start[0]
args.t_end=args.t_end[0]
args.dbscan_dist=args.dbscan_dist[0]
args.dbscan_nb=args.dbscan_nb[0]
args.cutoff_connect=args.cutoff_connect[0]
args.cluster_groups_file=args.cluster_groups_file[0]
args.nb_smoothing=args.nb_smoothing[0]

#show colour maps
#----------------
if args.show_colour_map:
	print ""
	print "The following standard matplotlib color maps can be used:"
	print ""
	print "Spectral, summer, coolwarm, pink_r, Set1, Set2, Set3, brg_r, Dark2, hot, PuOr_r, afmhot_r, terrain_r,"
	print "PuBuGn_r, RdPu, gist_ncar_r, gist_yarg_r, Dark2_r, YlGnBu, RdYlBu, hot_r, gist_rainbow_r, gist_stern, "
	print "gnuplot_r, cool_r, cool, gray, copper_r, Greens_r, GnBu, gist_ncar, spring_r, gist_rainbow, RdYlBu_r, "
	print "gist_heat_r, OrRd_r, CMRmap, bone, gist_stern_r, RdYlGn, Pastel2_r, spring, terrain, YlOrRd_r, Set2_r, "
	print "winter_r, PuBu, RdGy_r, spectral, flag_r, jet_r, RdPu_r, Purples_r, gist_yarg, BuGn, Paired_r, hsv_r, "
	print "bwr, cubehelix, YlOrRd, Greens, PRGn, gist_heat, spectral_r, Paired, hsv, Oranges_r, prism_r, Pastel2, "
	print "Pastel1_r, Pastel1, gray_r, PuRd_r, Spectral_r, gnuplot2_r, BuPu, YlGnBu_r, copper, gist_earth_r, "
	print "Set3_r, OrRd, PuBu_r, ocean_r, brg, gnuplot2, jet, bone_r, gist_earth, Oranges, RdYlGn_r, PiYG,"
	print "CMRmap_r, YlGn, binary_r, gist_gray_r, Accent, BuPu_r, gist_gray, flag, seismic_r, RdBu_r, BrBG, Reds,"
	print "BuGn_r, summer_r, GnBu_r, BrBG_r, Reds_r, RdGy, PuRd, Accent_r, Blues, Greys, autumn, cubehelix_r, "
	print "nipy_spectral_r, PRGn_r, Greys_r, pink, binary, winter, gnuplot, RdBu, prism, YlOrBr, coolwarm_r,"
	print "rainbow_r, rainbow, PiYG_r, YlGn_r, Blues_r, YlOrBr_r, seismic, Purples, bwr_r, autumn_r, ocean,"
	print "Set1_r, PuOr, PuBuGn, nipy_spectral, afmhot."
	print ""
	sys.exit(0)

#sanity check
#============
if not os.path.isfile(args.grofilename):
	print "Error: file " + str(args.grofilename) + " not found."
	sys.exit(1)
if args.cluster_groups_file!="no" and not os.path.isfile(args.cluster_groups_file):
	print "Error: file " + str(args.cluster_groups_file) + " not found."
	sys.exit(1)
if args.xtcfilename=="no":
	if '-t' in sys.argv:
		print "Error: -t option specified but no xtc file specified."
		sys.exit(1)
	elif '-b' in sys.argv:
		print "Error: -b option specified but no xtc file specified."
		sys.exit(1)
	elif '-e' in sys.argv:
		print "Error: -e option specified but no xtc file specified."
		sys.exit(1)
	elif '-w' in sys.argv:
		print "Error: -w option specified but no xtc file specified."
		sys.exit(1)
	elif '--smooth' in sys.argv:
		print "Error: --smooth option specified but no xtc file specified."
		sys.exit(1)
elif not os.path.isfile(args.xtcfilename):
	print "Error: file " + str(args.xtcfilename) + " not found."
	sys.exit(1)
if args.m_algorithm=="connectivity":
	if '--radius' in sys.argv:
		print "Error: --radius option specified but --algorithm option set to 'connectivity'."
		sys.exit(1)
	elif '--neighbours' in sys.argv:
		print "Error: --neighbours option specified but --algorithm option set to 'connectivity'."
		sys.exit(1)
else:
	if '--cutoff' in sys.argv:
		print "Error: --cutoff option specified but --algorithm option set to 'density'."
		sys.exit(1)

#create folders and log file
#===========================
if args.output_folder=="no":
	if args.xtcfilename=="no":
		args.output_folder="cluster_lip_" + args.grofilename[:-4]
	else:
		args.output_folder="cluster_lip_" + args.xtcfilename[:-4]
if os.path.isdir(args.output_folder):
	print "Error: folder " + str(args.output_folder) + " already exists, choose a different output name via -o."
	sys.exit(1)
else:
	#create folders
	#--------------
	os.mkdir(args.output_folder)
	#1 sizes
	os.mkdir(args.output_folder + "/1_sizes")
	if args.xtcfilename!="no":
		os.mkdir(args.output_folder + "/1_sizes/1_1_plots_2D")
		os.mkdir(args.output_folder + "/1_sizes/1_1_plots_2D/upper")
		os.mkdir(args.output_folder + "/1_sizes/1_1_plots_2D/upper/png")
		os.mkdir(args.output_folder + "/1_sizes/1_1_plots_2D/lower")
		os.mkdir(args.output_folder + "/1_sizes/1_1_plots_2D/lower/png")
	#2 groups
	if args.cluster_groups_file!="no":
		os.mkdir(args.output_folder + "/2_groups")
		if args.xtcfilename!="no":
			os.mkdir(args.output_folder + "/2_groups/2_1_plots_2D")
			os.mkdir(args.output_folder + "/2_groups/2_1_plots_2D/upper")
			os.mkdir(args.output_folder + "/2_groups/2_1_plots_2D/upper/png")
			os.mkdir(args.output_folder + "/2_groups/2_1_plots_2D/lower")
			os.mkdir(args.output_folder + "/2_groups/2_1_plots_2D/lower/png")
			os.mkdir(args.output_folder + "/2_groups/2_2_plots_1D")
			os.mkdir(args.output_folder + "/2_groups/2_2_plots_1D/png")
			os.mkdir(args.output_folder + "/2_groups/2_2_plots_1D/xvg")
			if args.nb_smoothing>1:
				os.mkdir(args.output_folder + "/2_groups/2_3_plots_1D_smoothed")
				os.mkdir(args.output_folder + "/2_groups/2_3_plots_1D_smoothed/png")
				os.mkdir(args.output_folder + "/2_groups/2_3_plots_1D_smoothed/xvg")	
	#3 snapshots
	os.mkdir(args.output_folder + "/3_snapshots")
	os.mkdir(args.output_folder + "/3_snapshots/sizes")
	if args.cluster_groups_file!="no":
		os.mkdir(args.output_folder + "/3_snapshots/groups")	
	#4 VMD
	if args.xtcfilename!="no":
		os.mkdir(args.output_folder + "/4_VMD")	
	
	#create log
	#----------
	filename_log=os.getcwd() + '/' + str(args.output_folder) + '/cluster_lip.log'
	output_log=open(filename_log, 'w')		
	output_log.write("[cluster_lip v" + str(version_nb) + "]\n")
	output_log.write("\nThis folder and its content were created using the following command:\n\n")
	tmp_log="python cluster_lip.py"
	for c in sys.argv[1:]:
		tmp_log+=" " + c
	output_log.write(tmp_log + "\n")
	output_log.close()
	#copy input files
	#----------------
	if args.cluster_groups_file!="no":
		shutil.copy2(args.cluster_groups_file,args.output_folder + "/")


################################################################################################################################################
# DICTIONARIES
################################################################################################################################################

#color maps dictionaries
colormaps_possible=['Spectral', 'summer', 'coolwarm', 'pink_r', 'Set1', 'Set2', 'Set3', 'brg_r', 'Dark2', 'hot', 'PuOr_r', 'afmhot_r', 'terrain_r', 'PuBuGn_r', 'RdPu', 'gist_ncar_r', 'gist_yarg_r', 'Dark2_r', 'YlGnBu', 'RdYlBu', 'hot_r', 'gist_rainbow_r', 'gist_stern', 'gnuplot_r', 'cool_r', 'cool', 'gray', 'copper_r', 'Greens_r', 'GnBu', 'gist_ncar', 'spring_r', 'gist_rainbow', 'RdYlBu_r', 'gist_heat_r', 'OrRd_r', 'CMRmap', 'bone', 'gist_stern_r', 'RdYlGn', 'Pastel2_r', 'spring', 'terrain', 'YlOrRd_r', 'Set2_r', 'winter_r', 'PuBu', 'RdGy_r', 'spectral', 'flag_r', 'jet_r', 'RdPu_r', 'Purples_r', 'gist_yarg', 'BuGn', 'Paired_r', 'hsv_r', 'bwr', 'cubehelix', 'YlOrRd', 'Greens', 'PRGn', 'gist_heat', 'spectral_r', 'Paired', 'hsv', 'Oranges_r', 'prism_r', 'Pastel2', 'Pastel1_r', 'Pastel1', 'gray_r', 'PuRd_r', 'Spectral_r', 'gnuplot2_r', 'BuPu', 'YlGnBu_r', 'copper', 'gist_earth_r', 'Set3_r', 'OrRd', 'PuBu_r', 'ocean_r', 'brg', 'gnuplot2', 'jet', 'bone_r', 'gist_earth', 'Oranges', 'RdYlGn_r', 'PiYG', 'CMRmap_r', 'YlGn', 'binary_r', 'gist_gray_r', 'Accent', 'BuPu_r', 'gist_gray', 'flag', 'seismic_r', 'RdBu_r', 'BrBG', 'Reds', 'BuGn_r', 'summer_r', 'GnBu_r', 'BrBG_r', 'Reds_r', 'RdGy', 'PuRd', 'Accent_r', 'Blues', 'Greys', 'autumn', 'cubehelix_r', 'nipy_spectral_r', 'PRGn_r', 'Greys_r', 'pink', 'binary', 'winter', 'gnuplot', 'RdBu', 'prism', 'YlOrBr', 'coolwarm_r', 'rainbow_r', 'rainbow', 'PiYG_r', 'YlGn_r', 'Blues_r', 'YlOrBr_r', 'seismic', 'Purples', 'bwr_r', 'autumn_r', 'ocean', 'Set1_r', 'PuOr', 'PuBuGn', 'nipy_spectral', 'afmhot']

#define leaflet selection string
leaflet_nb_lipids={}
leaflet_sele={}
leaflet_sele_string={}
leaflet_sele_string['martini']="name PO4 or name B1A or name PO3"
leaflet_sele_string['gromos']=""
leaflet_sele_string['charmm']=""

#define lipids taken into account
lipids_sele={}
lipids_sele_nb={}
lipids_sele_string={}
lipids_selection={}
lipids_selection_VMD_string={}
lipids_handled={}	#those dealt with
lipids_present={}	#those found
lipids_possible={}	#all the possible ones we can deal with
lipids_possible['martini']=['DHPC','DHPE','DLPC','DLPE','DAPC','DUPC','DPPC','DPPE','DPPS','DPPG','DSPC','DSPE','POPC','POPE','POPS','POPG','PPCS','PIP2','PIP3','PI3','GM3']

#case: martini
#-------------
if args.forcefield_opt=='martini':
	#PIPs
	for s in ['PIP2','PIP3','PI3']:
		lipids_sele_string[s]="resname " + str(s) + " and name PO3"
	#GM3s
	lipids_sele_string['GM3']="resname GM3 and name B1A"
	#usual lipids
	for s in ['DHPC','DHPE','DLPC','DLPE','DAPC','DUPC','DPPC','DPPE','DPPS','DPPG','DSPC','DSPE','POPC','POPE','POPS','POPG','PPCS']:
		lipids_sele_string[s]="resname " + str(s) + " and name PO4"
	
#case: gromos
#------------
elif args.forcefield_opt=='gromos':
	print "to do"
	sys.exit(1)

#case: charm
#-----------
elif args.forcefield_opt=='charmm':
	print "to do"
	sys.exit(1)

################################################################################################################################################
# DATA LOADING
################################################################################################################################################

# Create size groups
#===================
groups_nb=0
groups_sizes_dict={}
groups_boundaries=[]
groups_colors_nb=0
groups_colors_dict={}
groups_colors_list=[]
groups_colors_map="custom"
if args.cluster_groups_file!="no":
	#read group definition file
	print "\nReading cluster groups definition file..."
	with open(args.cluster_groups_file) as f:
		lines = f.readlines()
	groups_nb=len(lines)
	for g_index in range(0,groups_nb):
		l_content=lines[g_index].split(',')
		tmp_beg=int(l_content[0])
		tmp_end=l_content[1]
		groups_colors_dict[g_index]=l_content[2][:-1]					#[:-1] to get rid of the final '\n' character
		if tmp_end=="max":
			tmp_end=100000												#put a stupidly big size to cap the open ended group (might beed increasing for super-large systems...)
		else:
			tmp_end=int(tmp_end)
		groups_boundaries.append([tmp_beg,tmp_end])
		
	#check for boundaries overlapping
	prev_beg=groups_boundaries[0][0]
	prev_end=groups_boundaries[0][1]
	if prev_end<prev_beg:
		print "Error: the upper boundary is smaller than the lower boundary for specified cluster groups" + str(g) + "."
		sys.exit(1)
	for g in groups_boundaries[1:]:
		if g[1]<g[0]:
			print "Error: the upper boundary is smaller than the lower boundary for group " + str(g) + "."
			sys.exit(1)
		if g[0]<=prev_end:
			print "Error: specified cluster groups [" + str(prev_beg) + "," + str(prev_end) + "] and " + str(g) + " overlap or are not in increasing order."
			sys.exit(1)
		prev_beg=g[0]
		prev_end=g[1]
	
	#check if a custom color map has been specified or not
	if groups_nb>1 and len(numpy.unique(groups_colors_dict.values()))==1:
		if numpy.unique(groups_colors_dict.values())[0] in colormaps_possible:
			groups_colors_map=numpy.unique(groups_colors_dict.values())[0]
		else:
			print "Error: either the same color was specified for all groups or the color map '" + str(numpy.unique(groups_colors_dict.values())[0]) + "' is not valid."
			sys.exit(1)

	#create equivalency table between groups and sizes
	for g_index in range(0,groups_nb):
		bb=groups_boundaries[g_index]
		tmp_beg=bb[0]
		tmp_end=bb[1]
		for tmp_size in range(tmp_beg, tmp_end+1):
			groups_sizes_dict[tmp_size]=g_index
	for tmp_size in list(set(range(1,max(groups_sizes_dict.keys())))-set(groups_sizes_dict.keys())): 	#this handles potentially unaccounted for sizes up to the maximum specified by the user
		groups_sizes_dict[tmp_size]=groups_nb
	if max(groups_sizes_dict.keys())!=100000:															#this handles potentially unaccounted for sizes above the maximum specified by the user (in case it's not an open group)
		for tmp_size in range(max(groups_sizes_dict.keys())+1,100001):
			groups_sizes_dict[tmp_size]=groups_nb

	#display results
	print " -found " + str(groups_nb) + " cluster groups:"
	for g_index in range(0,groups_nb):
		if groups_boundaries[g_index][1]==100000:
			print "   g" + str(g_index) + "=" + str(groups_boundaries[g_index][0]) + "+, " + str(groups_colors_dict[g_index])
		else:
			print "   g" + str(g_index) + "=" + str(groups_boundaries[g_index][0]) + "-" + str(groups_boundaries[g_index][1]) + ", " + str(groups_colors_dict[g_index])
		

# Load universe
#==============
if args.xtcfilename=="no":
	print "\nLoading file..."
	U=Universe(args.grofilename)
	all_atoms=U.selectAtoms("all")
	nb_atoms=all_atoms.numberOfAtoms()
	nb_frames_xtc=1
	nb_frames_processed=1
else:
	print "\nLoading trajectory..."
	U=Universe(args.grofilename, args.xtcfilename)
	all_atoms=U.selectAtoms("all")
	nb_atoms=all_atoms.numberOfAtoms()
	nb_frames_xtc=U.trajectory.numframes
	nb_frames_processed=0
	U.trajectory.rewind()

# Identify leaflets
#==================
print "\nIdentifying leaflets..."
#identify lipids leaflet groups
if args.cutoff_leaflet:
	print " -optimising cutoff..."
	cutoff_value=MDAnalysis.analysis.leaflet.optimize_cutoff(U, leaflet_sele_string[args.forcefield_opt])
	L=MDAnalysis.analysis.leaflet.LeafletFinder(U, leaflet_sele_string[args.forcefield_opt], cutoff_value[0])
else:
	L=MDAnalysis.analysis.leaflet.LeafletFinder(U, leaflet_sele_string[args.forcefield_opt])
#process groups
if numpy.shape(L.groups())[0]<2:
	print "Error: imposssible to identify 2 leaflets."
	sys.exit(1)
else:
	if L.group(0).centerOfGeometry()[2] > L.group(1).centerOfGeometry()[2]:
		leaflet_sele["upper"]=L.group(0).residues.atoms
		leaflet_sele["lower"]=L.group(1).residues.atoms
	else:
		leaflet_sele["upper"]=L.group(1).residues.atoms
		leaflet_sele["lower"]=L.group(0).residues.atoms
	for l in ["lower","upper"]:
		leaflet_nb_lipids[l]=leaflet_sele[l].numberOfResidues()
	if numpy.shape(L.groups())[0]==2:
		print " -found 2 leaflets: ", leaflet_nb_lipids["upper"], ' (upper) and ', leaflet_nb_lipids["lower"], ' (lower) lipids'
	else:
		other_lipids=0
		for g in range(2, numpy.shape(L.groups())[0]):
			other_lipids+=L.group(g).numberOfResidues()
		print " -found " + str(numpy.shape(L.groups())[0]) + " groups: " + str(leaflet_nb_lipids["upper"]) + " (upper), " + str(leaflet_nb_lipids["lower"]) + " (lower) and " + str(other_lipids) + " (others) lipids respectively"

# Identify membrane composition
#==============================
print "\nIdentifying membrane composition..."

#store which lip resnames are present in each leaflet as well as those we can deal with
for l in ["lower","upper"]:
	lipids_present[l]=[]
	lipids_handled[l]=[]
	for s in numpy.unique(leaflet_sele[l].resnames()):
		lipids_present[l].append(s)
		if s in lipids_possible[args.forcefield_opt]:
			lipids_handled[l].append(s)
lipids_present["both"]=numpy.unique(lipids_present["lower"]+lipids_present["upper"])
lipids_handled["both"]=numpy.unique(lipids_handled["lower"]+lipids_handled["upper"])

#create particle selection for each lipid type
lipids_sele["all"]=MDAnalysis.core.AtomGroup.AtomGroup([])
for l in ["lower","upper"]:
	lipids_sele[l]={}
	lipids_sele_nb[l]={}
	lipids_selection[l]={}
	lipids_selection_VMD_string[l]={}
	for s in lipids_present[l]:
		lipids_selection[l][s]={}
		lipids_selection_VMD_string[l][s]={}
		lipids_sele[l][s]=leaflet_sele[l].selectAtoms(lipids_sele_string[s])
		lipids_sele_nb[l][s]=lipids_sele[l][s].numberOfResidues()
		lipids_sele["all"]+=lipids_sele[l][s].residues.atoms
		for r_index in range(0,lipids_sele_nb[l][s]):
			lipids_selection[l][s][r_index]=lipids_sele[l][s].selectAtoms("resnum " + str(lipids_sele[l][s].resnums()[r_index])).residues.atoms
			lipids_selection_VMD_string[l][s][r_index]="resname " + str(s) + " and resid " + str(lipids_sele[l][s].resnums()[r_index])

#specie ratios
lipids_ratio={}
lipids_ratio["lower"]={}
lipids_ratio["upper"]={}
membrane_comp={}
membrane_comp["lower"]=" -lower:"
membrane_comp["upper"]=" -upper:"
for l in ["lower","upper"]:
	for s in lipids_present[l]:
		lipids_ratio[l][s]=round(lipids_sele_nb[l][s]/float(leaflet_nb_lipids[l])*100,1)
		membrane_comp[l]+=" " + s + " (" + str(lipids_ratio[l][s]) + "%)"
print membrane_comp["upper"]
print membrane_comp["lower"]

################################################################################################################################################
# FUNCTIONS: algorithm
################################################################################################################################################

def detect_clusters_connectivity(loc_coords, box_dim):

	#get distances between the representative particle of the current lipid specie
	dist=MDAnalysis.analysis.distances.distance_array(numpy.float32(loc_coords), numpy.float32(loc_coords), box_dim)
	
	#use networkx algorithm
	connected=(dist<args.cutoff_connect)
	network=nx.Graph(connected)
	groups=nx.connected_components(network)
	
	return groups
def detect_clusters_density(loc_coords, box_dim):

	#get distances
	dist=MDAnalysis.analysis.distances.distance_array(numpy.float32(loc_coords), numpy.float32(loc_coords), box_dim)
	
	#run DBSCAN algorithm
	dbscan_output=DBSCAN(eps=args.dbscan_dist,metric='precomputed',min_samples=args.dbscan_nb).fit(dist)

	#build 'groups' structure i.e. a list whose element are all the clusters identified
	groups=[]
	for c_lab in numpy.unique(dbscan_output.labels_):
		tmp_pos=numpy.argwhere(dbscan_output.labels_==c_lab)
		if c_lab==-1:
			for p in tmp_pos:
				groups.append([p[0]])
		else:
			g=[]
			for p in tmp_pos:
				g.append(p[0])
			groups.append(g)

	return groups

################################################################################################################################################
# FUNCTIONS: calculate statistics
################################################################################################################################################

def rolling_avg(loc_list):
	
	loc_arr=numpy.asarray(loc_list)
	shape=(loc_arr.shape[-1]-args.nb_smoothing+1,args.nb_smoothing)
	strides=(loc_arr.strides[-1],loc_arr.strides[-1])   	
	return numpy.average(numpy.lib.stride_tricks.as_strided(loc_arr, shape=shape, strides=strides), -1)
def get_sizes_sampled():
	
	#sizes sampled
	#=============
	#case: gro file
	#--------------
	if args.xtcfilename=="no":
		for l in ["lower","upper"]:
			lipids_sizes_sampled[l]["all"]=[]
			for s in lipids_handled[l]:
				lipids_sizes_sampled[l][s]=lipids_cluster_size[l][s][0]
				lipids_sizes_sampled[l]["all"]=lipids_cluster_size[l][s][0]
				for r_index in range(1,lipids_sele_nb[l][s]):
					lipids_sizes_sampled[l][s]=list(numpy.unique(lipids_sizes_sampled[l][s] + lipids_cluster_size[l][s][r_index]))
				lipids_sizes_sampled[l]["all"]=list(numpy.unique(lipids_sizes_sampled[l]["all"] + lipids_sizes_sampled[l][s]))	
		lipids_sizes_sampled["both"]["all"]=list(numpy.unique(lipids_sizes_sampled["lower"]["all"] + lipids_sizes_sampled["upper"]["all"]))	
	
	#case: xtc file
	#--------------
	else:
		for l in ["lower","upper"]:
			for s in lipids_handled[l]:
				#sizes sampled by 1st lipid of that specie
				lipids_sizes_sampled[l][s]=list(numpy.unique(lipids_cluster_size[l][s][0]))
				lipids_sizes_sampled[l]["all"]=list(numpy.unique(lipids_cluster_size[l][s][0]))
				#sizes sampled by remainling lipids of that specie
				for r_index in range(1,lipids_sele_nb[l][s]):
					lipids_sizes_sampled[l][s]=list(numpy.unique(lipids_sizes_sampled[l][s] + list(numpy.unique(lipids_cluster_size[l][s][r_index]))))
				lipids_sizes_sampled[l]["all"]=list(numpy.unique(lipids_sizes_sampled[l]["all"] + lipids_sizes_sampled[l][s]))
		lipids_sizes_sampled["both"]["all"]=[]
		for s in lipids_handled["both"]:
			if s in lipids_handled["lower"] and s in lipids_handled["upper"]:
				lipids_sizes_sampled["both"][s]=list(numpy.unique(lipids_sizes_sampled["upper"][s]+lipids_sizes_sampled["lower"][s]))
			elif s in lipids_handled["lower"]:
				lipids_sizes_sampled["both"][s]=list(lipids_sizes_sampled["lower"][s])
			elif s in lipids_handled["upper"]:
				lipids_sizes_sampled["both"][s]=list(lipids_sizes_sampled["upper"][s])
			lipids_sizes_sampled["both"]["all"]=list(numpy.unique(lipids_sizes_sampled["both"]["all"] + lipids_sizes_sampled["both"][s]))
			
	#groups sampled
	#==============
	if args.cluster_groups_file!="no":
		#case: gro file
		#--------------
		if args.xtcfilename=="no":
			for l in ["lower","upper"]:
				lipids_groups_sampled[l]["all"]=[]
				for s in lipids_handled[l]:
					lipids_groups_sampled[l][s]=lipids_cluster_group[l][s][0]
					lipids_groups_sampled[l]["all"]=lipids_cluster_group[l][s][0]
					for r_index in range(1,lipids_sele_nb[l][s]):
						lipids_groups_sampled[l][s]=list(numpy.unique(lipids_groups_sampled[l][s] + lipids_cluster_group[l][s][r_index]))
					lipids_groups_sampled[l]["all"]=list(numpy.unique(lipids_groups_sampled[l]["all"] + lipids_groups_sampled[l][s]))	
			lipids_groups_sampled["both"]["all"]=list(numpy.unique(lipids_groups_sampled["lower"]["all"] + lipids_groups_sampled["upper"]["all"]))	
		
		#case: xtc file
		#--------------
		else:
			for l in ["lower","upper"]:
				for s in lipids_handled[l]:
					lipids_groups_sampled[l][s]=list(numpy.unique(lipids_cluster_group[l][s][0]))
					lipids_groups_sampled[l]["all"]=list(numpy.unique(lipids_cluster_group[l][s][0]))
					for r_index in range(1,lipids_sele_nb[l][s]):
						lipids_groups_sampled[l][s]=list(numpy.unique(lipids_groups_sampled[l][s] + list(numpy.unique(lipids_cluster_group[l][s][r_index]))))
					lipids_groups_sampled[l]["all"]=list(numpy.unique(lipids_groups_sampled[l]["all"] + lipids_groups_sampled[l][s]))
			lipids_groups_sampled["both"]["all"]=[]
			for s in lipids_handled["both"]:
				if s in lipids_handled["lower"] and s in lipids_handled["upper"]:
					lipids_groups_sampled["both"][s]=list(numpy.unique(lipids_groups_sampled["upper"][s]+lipids_groups_sampled["lower"][s]))
				elif s in lipids_handled["lower"]:
					lipids_groups_sampled["both"][s]=list(lipids_groups_sampled["lower"][s])
				elif s in lipids_handled["upper"]:
					lipids_groups_sampled["both"][s]=list(lipids_groups_sampled["upper"][s])
				lipids_groups_sampled["both"]["all"]=list(numpy.unique(lipids_groups_sampled["both"]["all"] + lipids_groups_sampled["both"][s]))

	return
def update_color_dict():
	
	global groups_colors_nb, groups_colors_list
	
	#NB: for groups the same color bar is used for all species
		
	#colormap for sizes: extract colors from jet colormap
	#-------------------
	sizes_colors_value=plt.cm.jet(numpy.linspace(0, 1, len(lipids_sizes_sampled["both"]["all"])))
	for s in lipids_handled["both"]:
		for c_size in lipids_sizes_sampled["both"][s]:
			c_index=lipids_sizes_sampled["both"]["all"].index(c_size)
			sizes_colors_dict[s][c_size]=sizes_colors_value[c_index]
		for k in sorted(sizes_colors_dict[s].iterkeys()):
			sizes_colors_list[s].append(sizes_colors_dict[s][k])
		sizes_colors_nb[s]=numpy.size(sizes_colors_dict[s].keys())

	#colormap for groups
	#-------------------
	if args.cluster_groups_file!="no":
		#case: user specified color map instead of colors
		if groups_colors_map!="custom":
			tmp_cmap=cm.get_cmap(groups_colors_map)
			groups_colors_value=tmp_cmap(numpy.linspace(0, 1, groups_nb))
			for g_index in range(0, groups_nb):
				groups_colors_dict[g_index]=groups_colors_value[g_index]

		#create list of colours ordered by group size
		groups_colors_nb=groups_nb
		for g in sorted(groups_colors_dict.iterkeys()):
			groups_colors_list.append(groups_colors_dict[g])

		#case: add the group 'other' in grey
		if groups_nb in lipids_groups_sampled["both"]["all"]:
			groups_colors_nb+=1
			groups_colors_dict[groups_nb]="#C0C0C0"
			groups_colors_list.append("#C0C0C0")						#choice of appending or prepending.. (top or bottom of colour bar...)
							
	return
def calc_stat():
		
	#preprocess: create dictionary of matrix of sizes sampled by each residue at each frame
	#--------------------------------------------------------------------------------------
	for l in ["lower","upper"]:
		for s in lipids_handled[l]:
			lipids_cluster_size_mat[l][s]=numpy.asarray(lipids_cluster_size[l][s].values())
			lipids_cluster_group_mat[l][s]=numpy.asarray(lipids_cluster_group[l][s].values())

	#preprocess: create data structure
	#---------------------------------
	for l in ["lower","upper"]:
		for s in lipids_handled[l]:
			for c_size in lipids_sizes_sampled[l][s]:
				sizes_pc[l][s][c_size]=[]
			for g_index in lipids_groups_sampled[l][s]:
				groups_pc[l][s][g_index]=[]

	#case: gro file
	#==============
	if args.xtcfilename=="no":
		tmp_groups_pc={}
		for l in ["lower","upper"]:
			for s in lipids_handled[l]:
				#initialise data
				for g_index in lipids_groups_sampled[l][s]:
					tmp_groups_pc[g_index]=0
				
				#sizes
				tmp_sizes=list(lipids_cluster_size_mat[l][s][:,0])
				for c_size in lipids_sizes_sampled[l][s]:
					#calculate % represented by size
					tmp_pc=tmp_sizes.count(c_size)/float(lipids_sele_nb[l][s])*100
					sizes_pc[l][s][c_size].append(tmp_pc)
											
					#add to group to which the size belong to
					if args.cluster_groups_file!="no":
						tmp_groups_pc[groups_sizes_dict[c_size]]+=tmp_pc
				
				#groups
				if args.cluster_groups_file!="no":
					for g_index in lipids_groups_sampled[l][s]:
						#calculate % represented by size
						groups_pc[l][s][g_index].append(tmp_groups_pc[g_index])

	#case: xtc file
	#==============
	else:			
		#calculate % accounted for by each cluster size / group
		#------------------------------------------------------
		frame_counter=0
		for frame in sorted(time_stamp.iterkeys()):
			frame_index=sorted(time_stamp.keys()).index(frame)
			frame_counter+=1
			#update progress
			progress='\r -processing frame ' + str(frame_counter) + '/' + str(nb_frames_processed) + '        '
			sys.stdout.flush()
			sys.stdout.write(progress)
			
			#browse lipids
			tmp_groups_pc={}
			for l in ["lower","upper"]:
				for s in lipids_handled[l]:
					#initialise data
					for g_index in lipids_groups_sampled[l][s]:
						tmp_groups_pc[g_index]=0
					
					#sizes
					tmp_sizes=list(lipids_cluster_size_mat[l][s][:,frame_index])
					for c_size in lipids_sizes_sampled[l][s]:
						#calculate % represented by size
						tmp_pc=tmp_sizes.count(c_size)/float(lipids_sele_nb[l][s])*100
						sizes_pc[l][s][c_size].append(tmp_pc)
												
						#add to group to which the size belong to
						if args.cluster_groups_file!="no":
							tmp_groups_pc[groups_sizes_dict[c_size]]+=tmp_pc
					
					#groups
					if args.cluster_groups_file!="no":
						for g_index in lipids_groups_sampled[l][s]:
							#calculate % represented by size
							groups_pc[l][s][g_index].append(tmp_groups_pc[g_index])
						
		#calculate longest stability of each group
		#-----------------------------------------
		if args.cluster_groups_file!="no":
			for l in ["lower","upper"]:
				for s in lipids_handled[l]:
					for g_index in lipids_groups_sampled["both"]["all"]:
						#find max stability of current group index for each lipid
						tmp_lipids_stability={}
						for r_index in range(0,lipids_sele_nb[l][s]):
							if g_index in lipids_cluster_group_mat[l][s][r_index,:]:
								tmp_lipids_stability[r_index]=max(len(list(v)) for g,v in itertools.groupby(lipids_cluster_group_mat[l][s][r_index,:], lambda x: x == g_index) if g)
							else:
								tmp_lipids_stability[r_index]=0
						#store maximum stability
						groups_stability[l][s][g_index]=max(tmp_lipids_stability.values())
		print ""
	
	return
def smooth_data():

	global time_sorted, time_smoothed
	
	#sort data into ordered lists
	#----------------------------
	for frame in sorted(time_stamp.keys()):
		time_sorted.append(time_stamp[frame])
		
	#calculate running average on sorted lists
	#-----------------------------------------
	if args.nb_smoothing>1:
		#time
		time_smoothed=rolling_avg(time_sorted)		
		
		#groups
		for l in ["lower","upper"]:
			for s in lipids_handled[l]:
				for g_index in lipids_groups_sampled[l][s]:
					groups_pc_smoothed[l][s][g_index]=rolling_avg(groups_pc[l][s][g_index])	
	
	return

################################################################################################################################################
# FUNCTIONS: write outputs
################################################################################################################################################

def write_warning():
	filename_details=os.getcwd() + '/' + str(args.output_folder) + '/warning.stat'
	output_stat = open(filename_details, 'w')		
	output_stat.write("[protein clustering statistics - written by cluster_prot v" + str(version_nb) + "]\n")
	output_stat.write("\n")	
	#general info
	output_stat.write("1. Nb of proteins: " + str(proteins_nb) + "\n")
	output_stat.write("2. Cluster detection Method:\n")
	if args.m_algorithm=='connectivity':
		output_stat.write(" - connectivity based (min distances)\n")
		output_stat.write(" - contact cutoff = " + str(args.cutoff_connect) + " Angstrom\n")
	elif args.m_algorithm=='density':
		output_stat.write(" - connectivity based (cog distances)\n")
		output_stat.write(" - contact cutoff = " + str(args.cutoff_connect) + " Angstrom\n")
	else:
		output_stat.write(" - density based (DBSCAN)\n")
		output_stat.write(" - search radius = " + str(args.dbscan_dist) + " Angstrom, nb of neighbours = " + str(args.dbscan_nb) + "\n")
	output_stat.write("\n")
	#warning message
	output_stat.write("Warning: a single cluster size (" + str(proteins_sizes_sampled[0]) + ") was detected throughout the trajectory. Check the -m, -c, -r or -n options (see cluster_prot -h).")
	output_stat.close()
	
	return

#case: xtc file
#==============
def graph_aggregation_2D_sizes():
	
	for l in ["lower","upper"]:
		for s in lipids_handled[l]:	
			#create filenames
			filename_png=os.getcwd() + '/' + str(args.output_folder) + '/1_sizes/1_1_plots_2D/' + str(l) + '/png/1_1_clusterlip_2D_' + str(l) + '_' + str(s) + '.png'
			filename_svg=os.getcwd() + '/' + str(args.output_folder) + '/1_sizes/1_1_plots_2D/' + str(l) + '/1_1_clusterlip_2D_' + str(l) + '_' + str(s) + '.svg'

			#create data
			lip_2D_evolution=numpy.zeros((lipids_sele_nb[l][s],len(time_stamp.keys())))
			for r_index in range(0,lipids_sele_nb[l][s]):
				lip_2D_evolution[r_index,:]=numpy.asarray(lipids_cluster_size[l][s][r_index])
	
			#build color map
			color_map=mcolors.LinearSegmentedColormap.from_list('custom', sizes_colors_list[s], sizes_colors_nb[s])
			
			#determine nb of colours and their boundaries
			bounds=[]
			cb_ticks_lab=[]
			for c in sorted(sizes_colors_dict[s].iterkeys()):
				bounds.append(c-0.5)
				cb_ticks_lab.append(str(c))
			bounds.append(sorted(sizes_colors_dict[s].iterkeys())[-1]+0.5)
			norm=mpl.colors.BoundaryNorm(bounds, color_map.N)
						
			#create figure ('norm' requires at least 2 elements to work)
			fig=plt.figure(figsize=(9, 8))
			ax_plot=fig.add_axes([0.10, 0.1, 0.75, 0.77])	
			ax_plot.matshow(lip_2D_evolution, origin='lower', interpolation='nearest', cmap=color_map, aspect='auto', norm=norm)

			#create color bar
			ax_cbar=fig.add_axes([0.88, 0.1, 0.025, 0.77])
			cb=mpl.colorbar.ColorbarBase(ax_cbar, cmap=color_map, norm=norm, boundaries=bounds)
		
			#position and label color bar ticks
			cb_ticks_pos=[]
			for b in range(1,len(bounds)):
				cb_ticks_pos.append(bounds[b-1]+(bounds[b]-bounds[b-1])/2)
			cb_ticks_pos.append(bounds[-1])
			cb.set_ticks(cb_ticks_pos)
			cb.set_ticklabels(cb_ticks_lab)
			for t in cb.ax.get_yticklabels():
				t.set_fontsize('xx-small')
			
			#x axis ticks
			ax_plot.xaxis.set_label_position('bottom') 
			ax_plot.xaxis.set_ticks_position('bottom')
			xticks_pos=ax_plot.xaxis.get_ticklocs()[1:-1]
			tmp_xticks_lab=[""]
			for t in sorted(time_stamp.values()):
				tmp_xticks_lab.append('{0:0g}'.format(numpy.floor(t)))
			xticks_lab=[""]
			for t in xticks_pos:
				xticks_lab.append(tmp_xticks_lab[int(t)+1])
			ax_plot.xaxis.set_ticklabels(xticks_lab)
		
			#y axis ticks (increase the index by 1 to get 1-based numbers)
			ax_plot.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x,p: '{0:0g}'.format(x+1)))
			
			#set title and limits
			ax_plot.set_xlabel("time (ns)", fontsize="medium")
			ax_plot.set_ylabel("lipid #", fontsize="medium")
			plt.setp(ax_plot.xaxis.get_majorticklabels(), fontsize="small" )
			plt.setp(ax_plot.yaxis.get_majorticklabels(), fontsize="small" )
			ax_plot.yaxis.set_major_locator(MaxNLocator(prune='lower'))	
			ax_plot.set_title("Evolution of the cluster size in which " + str(s) + " lipids are involved", fontsize="medium")	
			ax_cbar.set_ylabel('cluster size',fontsize='small')
			
			#save figure
			fig.savefig(filename_png)
			fig.savefig(filename_svg)
			plt.close()
			
	return
def graph_aggregation_2D_groups():
	
	for l in ["lower","upper"]:
		for s in lipids_handled[l]:	
			#create filenames
			filename_png=os.getcwd() + '/' + str(args.output_folder) + '/2_groups/2_1_plots_2D/' + str(l) + '/png/2_1_clusterlip_2D_' + str(l) + '_' + str(s) + '.png'
			filename_svg=os.getcwd() + '/' + str(args.output_folder) + '/2_groups/2_1_plots_2D/' + str(l) + '/2_1_clusterlip_2D_' + str(l) + '_' + str(s) + '.svg'

			#create data
			lip_2D_evolution=numpy.zeros((lipids_sele_nb[l][s],len(time_stamp.keys())))
			for r_index in range(0,lipids_sele_nb[l][s]):
				lip_2D_evolution[r_index,:]=numpy.asarray(lipids_cluster_group[l][s][r_index])
	
			#build color map
			color_map=mcolors.LinearSegmentedColormap.from_list('custom', groups_colors_list, groups_colors_nb)
			
			#determine nb of colours and their boundaries
			bounds=[]
			cb_ticks_lab=[]
			for g_index in sorted(groups_colors_dict.iterkeys()):
				bounds.append(g_index-0.5)
				if g_index==groups_nb:
					cb_ticks_lab.append("other")
				elif groups_boundaries[g_index][1]==100000:
					cb_ticks_lab.append(">=" + str(groups_boundaries[g_index][0]))
				else:
					cb_ticks_lab.append(str(groups_boundaries[g_index][0]) + "-" + str(groups_boundaries[g_index][1]))
			bounds.append(sorted(groups_colors_dict.iterkeys())[-1]+0.5)
			norm=mpl.colors.BoundaryNorm(bounds, color_map.N)
						
			#create figure ('norm' requires at least 2 elements to work)
			fig=plt.figure(figsize=(9, 8))
			ax_plot=fig.add_axes([0.09, 0.1, 0.75, 0.77])	
			ax_plot.matshow(lip_2D_evolution, origin='lower', interpolation='nearest', cmap=color_map, aspect='auto', norm=norm)

			#create color bar
			ax_cbar=fig.add_axes([0.87, 0.1, 0.025, 0.77])
			cb=mpl.colorbar.ColorbarBase(ax_cbar, cmap=color_map, norm=norm, boundaries=bounds)
		
			#position and label color bar ticks
			cb_ticks_pos=[]
			for b in range(1,len(bounds)):
				cb_ticks_pos.append(bounds[b-1]+(bounds[b]-bounds[b-1])/2)
			cb_ticks_pos.append(bounds[-1])
			cb.set_ticks(cb_ticks_pos)
			cb.set_ticklabels(cb_ticks_lab)
			for t in cb.ax.get_yticklabels():
				t.set_fontsize('small')
						
			#x axis ticks
			ax_plot.xaxis.set_label_position('bottom') 
			ax_plot.xaxis.set_ticks_position('bottom')
			xticks_pos=ax_plot.xaxis.get_ticklocs()[1:-1]
			tmp_xticks_lab=[""]
			for t in sorted(time_stamp.values()):
				tmp_xticks_lab.append('{0:0g}'.format(numpy.floor(t)))
			xticks_lab=[""]
			for t in xticks_pos:
				xticks_lab.append(tmp_xticks_lab[int(t)+1])
			ax_plot.xaxis.set_ticklabels(xticks_lab)
		
			#y axis ticks (increase the index by 1 to get 1-based numbers)
			ax_plot.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x,p: '{0:0g}'.format(x+1)))
			
			#set title and limits
			ax_plot.set_xlabel("time (ns)", fontsize="medium")
			ax_plot.set_ylabel("lipid #", fontsize="medium")
			plt.setp(ax_plot.xaxis.get_majorticklabels(), fontsize="small" )
			plt.setp(ax_plot.yaxis.get_majorticklabels(), fontsize="small" )
			ax_plot.yaxis.set_major_locator(MaxNLocator(prune='lower'))	
			ax_plot.set_title("Evolution of the cluster size group in which " + str(s) + " lipids are involved", fontsize="medium")	
			ax_cbar.set_ylabel('cluster size',fontsize='medium')
			
			#save figure
			fig.savefig(filename_png)
			fig.savefig(filename_svg)
			plt.close()
			
	return
def write_stability_groups():
	
	filename_details=os.getcwd() + '/' + str(args.output_folder) + '/2_groups/2_0_clusterlip_stability.stat'
	output_stat = open(filename_details, 'w')		
	output_stat.write("[lipid clustering statistics - written by cluster_lip v" + str(version_nb) + "]\n")
	output_stat.write("\n")

	#general info
	output_stat.write("1. Membrane compositionL\n")
	output_stat.write(membrane_comp["upper"] + "\n")
	output_stat.write(membrane_comp["lower"] + "\n")
	tmp_string=str(lipids_handled["both"][0])
	for s in lipids_handled["both"]:
		tmp_string+=", " + str(s)
	output_stat.write("\n")
	output_stat.write("2. Lipid species processed: " + str(tmp_string) + "\n")
	output_stat.write("\n")
	output_stat.write("3. Cluster detection Method:\n")
	if args.m_algorithm=='connectivity':
		output_stat.write(" - connectivity based\n")
		output_stat.write(" - contact cutoff = " + str(args.cutoff_connect) + " Angstrom\n")
	else:
		output_stat.write(" - density based (DBSCAN)\n")
		output_stat.write(" - search radius = " + str(args.dbscan_dist) + " Angstrom, nb of neighbours = " + str(args.dbscan_nb) + "\n")
	output_stat.write("\n")
	output_stat.write("Maximum stability (in number of consecutive frames) for each cluster groups\n")
	output_stat.write("Note: frames skipped are not taken into account (the nb below correspond to consecutive frames *processed*)\n")
	
	#title bars
	#==========
	tmp_cap1=""
	tmp_cap2="-----"
	for g_index in lipids_groups_sampled["both"]["all"]:
		if g_index==groups_nb:
			tmp_cap1+="	other"
		elif groups_boundaries[g_index][1]==100000:
			tmp_cap1+="	>=" + str(groups_boundaries[g_index][0])
		else:
			tmp_cap1+="	" + str(groups_boundaries[g_index][0]) + "-" + str(groups_boundaries[g_index][1])
		tmp_cap2+="--------"

	#upper leaflet
	#=============
	output_stat.write("\n")
	output_stat.write("upper leaflet\n")
	output_stat.write("=============\n")
	output_stat.write(tmp_cap1 + "\n")
	output_stat.write(tmp_cap2 + "\n")
	for s in lipids_handled["upper"]:
		results=str(s)
		for g_index in lipids_groups_sampled["both"]["all"]:
			results+= "	" + str(groups_stability["upper"][s][g_index])
		output_stat.write(results + "\n")

	#lower leaflet
	#=============
	output_stat.write("\n")
	output_stat.write("lower leaflet\n")
	output_stat.write("=============\n")
	output_stat.write(tmp_cap1 + "\n")
	output_stat.write(tmp_cap2 + "\n")
	for s in lipids_handled["lower"]:
		results=str(s)
		for g_index in lipids_groups_sampled["both"]["all"]:
			results+= "	" + str(groups_stability["lower"][s][g_index])
		output_stat.write(results + "\n")

	output_stat.close()
	
	return
def write_xvg_groups():
	
	for l in ["lower","upper"]:
		for s in lipids_handled[l]:
			filename_txt=os.getcwd() + '/' + str(args.output_folder) + '/2_groups/2_2_plots_1D/xvg/2_2_clusterlip_1D_' + str(l) + '_' + str(s) + '.txt'
			filename_xvg=os.getcwd() + '/' + str(args.output_folder) + '/2_groups/2_2_plots_1D/xvg/2_2_clusterlip_1D_' + str(l) + '_' + str(s) + '.xvg'
			output_txt = open(filename_txt, 'w')
			output_txt.write("@[lipid tail order parameters statistics - written by cluster_lip v" + str(version_nb) + "]\n")
			output_txt.write("@Use this file as the argument of the -c option of the script 'xvg_animate' in order to make a time lapse movie of the data in 2_2_clusterlip_1D_" + str(l) + "_" + str(s) + ".xvg\n")
			output_xvg = open(filename_xvg, 'w')
			output_xvg.write("@ title \"Number of lipid clusters\"\n")
			output_xvg.write("@ xaxis  label \"time (ns)\"\n")
			output_xvg.write("@ autoscale ONREAD xaxes\n")
			output_xvg.write("@ TYPE XY\n")
			output_xvg.write("@ view 0.15, 0.15, 0.95, 0.85\n")
			output_xvg.write("@ legend on\n")
			output_xvg.write("@ legend box on\n")
			output_xvg.write("@ legend loctype view\n")
			output_xvg.write("@ legend 0.98, 0.8\n")
			output_xvg.write("@ legend length " + str(len(lipids_groups_sampled[l][s])) + "\n")
			for g in range(0,len(lipids_groups_sampled[l][s])):
				g_index=lipids_groups_sampled[l][s][g]
				if g_index==groups_nb:
					output_xvg.write("@ s" + str(g) + " legend \"other\"\n")
					output_txt.write("2_2_clusterlip_1D_" + str(l) + "_" + str(s) +".xvg," + str(g+1) + ",other," + mcolors.rgb2hex(groups_colors_dict[g_index]) + "\n")
				elif groups_boundaries[g_index][1]==100000:
					output_xvg.write("@ s" + str(g) + " legend \">=" + str(groups_boundaries[g_index][0]) + "\"\n")
					output_txt.write("2_2_clusterlip_1D_" + str(l) + "_" + str(s) +".xvg," + str(g+1) + ",>=" + str(groups_boundaries[g_index][0]) + "," + mcolors.rgb2hex(groups_colors_dict[g_index]) + "\n")
				else:
					output_xvg.write("@ s" + str(g) + " legend \"" + str(groups_boundaries[g_index][0]) + "-" + str(groups_boundaries[g_index][1]) + "\"\n")
					output_txt.write("2_2_clusterlip_1D_" + str(l) + "_" + str(s) +".xvg," + str(g+1) + "," + str(groups_boundaries[g_index][0]) + "-" + str(groups_boundaries[g_index][1]) + "," + mcolors.rgb2hex(groups_colors_dict[g_index]) + "\n")
			output_txt.close()
			for frame in sorted(time_stamp.iterkeys()):
				results=str(time_stamp[frame])
				frame_index=sorted(time_stamp.keys()).index(frame)
				for g_index in lipids_groups_sampled[l][s]:
					results+="	" + str(round(groups_pc[l][s][g_index][frame_index],2))
				output_xvg.write(results + "\n")
			output_xvg.close()
	
	return
def graph_xvg_groups():
	
	#create filenames
	#----------------
	for s in lipids_handled["both"]:
		filename_png=os.getcwd() + '/' + str(args.output_folder) + '/2_groups/2_2_plots_1D/png/2_2_clusterlip_1D_' + str(s) + '.png'
		filename_svg=os.getcwd() + '/' + str(args.output_folder) + '/2_groups/2_2_plots_1D/2_2_clusterlip_1D_' + str(s) + '.svg'

		#create figure
		#-------------
		fig=plt.figure(figsize=(8, 6.2))
		fig.suptitle("Evolution of " + str(s) + " lipid distribution")
			
		#plot data: upper leafet
		#-----------------------
		ax1 = fig.add_subplot(211)
		p_upper={}
		if s in lipids_handled["upper"]:
			for g_index in lipids_groups_sampled["upper"][s]:
				if g_index==groups_nb:
					tmp_label="other"
				elif groups_boundaries[g_index][1]==100000:
					tmp_label=">=" + str(groups_boundaries[g_index][0])
				else:
					tmp_label=str(groups_boundaries[g_index][0]) + "-" + str(groups_boundaries[g_index][1])
				p_upper[g_index]=plt.plot(time_sorted, groups_pc["upper"][s][g_index], color=groups_colors_dict[g_index], linewidth=2.0, label=tmp_label)
			fontP.set_size("small")
			ax1.legend(prop=fontP)
		plt.title("upper leaflet", fontsize="small")
		plt.xlabel('time (ns)', fontsize="small")
		plt.ylabel('% of lipids', fontsize="small")

		#plot data: lower leafet
		#-----------------------
		ax2 = fig.add_subplot(212)
		p_lower={}
		if s in lipids_handled["lower"]:
			for g_index in lipids_groups_sampled["lower"][s]:
				if g_index==groups_nb:
					tmp_label="other"
				elif groups_boundaries[g_index][1]==100000:
					tmp_label=">=" + str(groups_boundaries[g_index][0])
				else:
					tmp_label=str(groups_boundaries[g_index][0]) + "-" + str(groups_boundaries[g_index][1])
				p_lower[g_index]=plt.plot(time_sorted, groups_pc["lower"][s][g_index], color=groups_colors_dict[g_index], linewidth=2.0, label=tmp_label)
			fontP.set_size("small")
			ax2.legend(prop=fontP)
		plt.title("lower leaflet", fontsize="small")
		plt.xlabel('time (ns)', fontsize="small")
		plt.ylabel('% of lipids', fontsize="small")
	
		#save figure
		#-----------
		ax1.set_ylim(0, 100)
		ax2.set_ylim(0, 100)
		ax1.xaxis.set_major_locator(MaxNLocator(nbins=5))
		ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
		ax2.xaxis.set_major_locator(MaxNLocator(nbins=5))
		ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))
		plt.setp(ax1.xaxis.get_majorticklabels(), fontsize="small" )
		plt.setp(ax1.yaxis.get_majorticklabels(), fontsize="small" )
		plt.setp(ax2.xaxis.get_majorticklabels(), fontsize="small" )
		plt.setp(ax2.yaxis.get_majorticklabels(), fontsize="small" )	
		plt.subplots_adjust(top=0.9, bottom=0.07, hspace=0.37, left=0.09, right=0.96)
		fig.savefig(filename_png)
		fig.savefig(filename_svg)
		plt.close()
	
	return
def write_xvg_groups_smoothed():
	for l in ["lower","upper"]:
		for s in lipids_handled[l]:
			filename_txt=os.getcwd() + '/' + str(args.output_folder) + '/2_groups/2_3_plots_1D_smoothed/xvg/2_3_clusterlip_1D_' + str(l) + '_' + str(s) + '_smoothed.txt'
			filename_xvg=os.getcwd() + '/' + str(args.output_folder) + '/2_groups/2_3_plots_1D_smoothed/xvg/2_3_clusterlip_1D_' + str(l) + '_' + str(s) + '_smoothed.xvg'
			output_txt = open(filename_txt, 'w')
			output_txt.write("@[lipid tail order parameters statistics - written by cluster_lip v" + str(version_nb) + "]\n")
			output_txt.write("@Use this file as the argument of the -c option of the script 'xvg_animate' in order to make a time lapse movie of the data in 2_3_clusterlip_1D_" + str(l) + "_" + str(s) + "_smoothed.xvg\n")
			output_xvg = open(filename_xvg, 'w')
			output_xvg.write("@ title \"Number of lipid clusters\"\n")
			output_xvg.write("@ xaxis  label \"time (ns)\"\n")
			output_xvg.write("@ autoscale ONREAD xaxes\n")
			output_xvg.write("@ TYPE XY\n")
			output_xvg.write("@ view 0.15, 0.15, 0.95, 0.85\n")
			output_xvg.write("@ legend on\n")
			output_xvg.write("@ legend box on\n")
			output_xvg.write("@ legend loctype view\n")
			output_xvg.write("@ legend 0.98, 0.8\n")
			output_xvg.write("@ legend length " + str(len(lipids_groups_sampled[l][s])) + "\n")
			for g in range(0,len(lipids_groups_sampled[l][s])):
				g_index=lipids_groups_sampled[l][s][g]
				if g==groups_nb:
					output_xvg.write("@ s" + str(g) + " legend \"other\"\n")
					output_txt.write("2_3_clusterlip_1D_" + str(l) + "_" + str(s) +"_smoothed.xvg," + str(g+1) + ",other," + mcolors.rgb2hex(groups_colors_dict[g_index]) + "\n")
				elif groups_boundaries[g_index][1]==100000:
					output_xvg.write("@ s" + str(g) + " legend \">=" + str(groups_boundaries[g_index][0]) + "\"\n")
					output_txt.write("2_3_clusterlip_1D_" + str(l) + "_" + str(s) +"_smoothed.xvg," + str(g+1) + ",>=" + str(groups_boundaries[g_index][0]) + "," + mcolors.rgb2hex(groups_colors_dict[g_index]) + "\n")
				else:
					output_xvg.write("@ s" + str(g) + " legend \"" + str(groups_boundaries[g_index][0]) + "-" + str(groups_boundaries[g_index][1]) + "\"\n")
					output_txt.write("2_3_clusterlip_1D_" + str(l) + "_" + str(s) +"_smoothed.xvg," + str(g+1) + "," + str(groups_boundaries[g_index][0]) + "-" + str(groups_boundaries[g_index][1]) + "," + mcolors.rgb2hex(groups_colors_dict[g_index]) + "\n")
			output_txt.close()
			for frame_index in range(0, len(time_smoothed)):
				results=str(time_smoothed[frame_index])
				for g_index in lipids_groups_sampled[l][s]:
					results+="	" + str(round(groups_pc_smoothed[l][s][g_index][frame_index],2))
				output_xvg.write(results + "\n")
			output_xvg.close()

	return
def graph_xvg_groups_smoothed():
	
	#create filenames
	#----------------
	for s in lipids_handled["both"]:
		filename_png=os.getcwd() + '/' + str(args.output_folder) + '/2_groups/2_3_plots_1D_smoothed/png/2_3_clusterlip_1D_' + str(s) + '_smoothed.png'
		filename_svg=os.getcwd() + '/' + str(args.output_folder) + '/2_groups/2_3_plots_1D_smoothed/2_3_clusterlip_1D_' + str(s) + '_smoothed.svg'

		#create figure
		#-------------
		fig=plt.figure(figsize=(8, 6.2))
		fig.suptitle("Evolution of " + str(s) + " lipid distribution")
			
		#plot data: upper leafet
		#-----------------------
		ax1 = fig.add_subplot(211)
		p_upper={}
		if s in lipids_handled["upper"]:
			for g_index in lipids_groups_sampled["upper"][s]:
				if g_index==groups_nb:
					tmp_label="other"
				elif groups_boundaries[g_index][1]==100000:
					tmp_label=">=" + str(groups_boundaries[g_index][0])
				else:
					tmp_label=str(groups_boundaries[g_index][0]) + "-" + str(groups_boundaries[g_index][1])
				p_upper[g_index]=plt.plot(time_smoothed, groups_pc_smoothed["upper"][s][g_index], color=groups_colors_dict[g_index], linewidth=2.0, label=tmp_label)
			fontP.set_size("small")
			ax1.legend(prop=fontP)
		plt.title("upper leaflet", fontsize="small")
		plt.xlabel('time (ns)', fontsize="small")
		plt.ylabel('% of lipids', fontsize="small")

		#plot data: lower leafet
		#-----------------------
		ax2 = fig.add_subplot(212)
		p_lower={}
		if s in lipids_handled["lower"]:
			for g_index in lipids_groups_sampled["lower"][s]:
				if g_index==groups_nb:
					tmp_label="other"
				elif groups_boundaries[g_index][1]==100000:
					tmp_label=">=" + str(groups_boundaries[g_index][0])
				else:
					tmp_label=str(groups_boundaries[g_index][0]) + "-" + str(groups_boundaries[g_index][1])
				p_lower[g_index]=plt.plot(time_smoothed, groups_pc_smoothed["lower"][s][g_index], color=groups_colors_dict[g_index], linewidth=2.0, label=tmp_label)
			fontP.set_size("small")
			ax2.legend(prop=fontP)
		plt.title("lower leaflet", fontsize="small")
		plt.xlabel('time (ns)', fontsize="small")
		plt.ylabel('% of lipids', fontsize="small")
	
		#save figure
		#-----------
		ax1.set_ylim(0, 100)
		ax2.set_ylim(0, 100)
		ax1.xaxis.set_major_locator(MaxNLocator(nbins=5))
		ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
		ax2.xaxis.set_major_locator(MaxNLocator(nbins=5))
		ax2.yaxis.set_major_locator(MaxNLocator(nbins=5))
		plt.setp(ax1.xaxis.get_majorticklabels(), fontsize="small" )
		plt.setp(ax1.yaxis.get_majorticklabels(), fontsize="small" )
		plt.setp(ax2.xaxis.get_majorticklabels(), fontsize="small" )
		plt.setp(ax2.yaxis.get_majorticklabels(), fontsize="small" )	
		plt.subplots_adjust(top=0.9, bottom=0.07, hspace=0.37, left=0.09, right=0.96)
		fig.savefig(filename_png)
		fig.savefig(filename_svg)
		plt.close()

	return

#annotations
#===========
def write_frame_stat(f_nb, f_index, t):
	
	#case: gro file or xtc summary
	#=============================
	if f_index=="all" and t=="all":
		#sizes
		#-----
		#create file
		if args.xtcfilename=="no":
			if args.m_algorithm=='connectivity':
				tmp_name=os.getcwd() + "/" + str(args.output_folder) + '/1_sizes/1_0_' + args.grofilename[:-4] + '_annotated_clusterlip_' + str(args.m_algorithm) + '_c' + str(int(args.cutoff_connect)) + '_sizes_sampled.stat'
			else:
				tmp_name=os.getcwd() + "/" + str(args.output_folder) + '/1_sizes/1_0_' + args.grofilename[:-4] + '_annotated_clusterlip_' + str(args.m_algorithm) + '_r' + str(int(args.dbscan_dist)) + '_n' + str(args.dbscan_nb) + '_sizes_sampled.stat'
		else:
			if args.m_algorithm=='connectivity':
				tmp_name=os.getcwd() + "/" + str(args.output_folder) + '/1_sizes/1_0_' + args.xtcfilename[:-4] + '_annotated_clusterlip_' + str(args.m_algorithm) + '_c' + str(int(args.cutoff_connect)) + '_sizes_sampled.stat'
			else:
				tmp_name=os.getcwd() + "/" + str(args.output_folder) + '/1_sizes/1_0_' + args.xtcfilename[:-4] + '_annotated_clusterlip_' + str(args.m_algorithm) + '_r' + str(int(args.dbscan_dist)) + '_n' + str(args.dbscan_nb) + '_sizes_sampled.stat'
		output_stat = open(tmp_name, 'w')
		
		#general info
		output_stat.write("[lipid clustering statistics - written by cluster_lip v" + str(version_nb) + "]\n")
		output_stat.write("\n")
		output_stat.write("1. membrane composition:\n")
		output_stat.write(membrane_comp["upper"] + "\n")
		output_stat.write(membrane_comp["lower"] + "\n")
		tmp_string=str(lipids_handled["both"][0])
		for s in lipids_handled["both"][1:]:
			tmp_string+=", " + str(s)
		output_stat.write("\n")
		output_stat.write("2. lipid species processed: " + str(tmp_string) + "\n")
		output_stat.write("\n")
		output_stat.write("3. cluster detection Method:\n")
		if args.m_algorithm=='connectivity':
			output_stat.write(" - connectivity based\n")
			output_stat.write(" - contact cutoff = " + str(args.cutoff_connect) + " Angstrom\n")
		else:
			output_stat.write(" - density based (DBSCAN)\n")
			output_stat.write(" - search radius = " + str(args.dbscan_dist) + " Angstrom, nb of neighbours = " + str(args.dbscan_nb) + "\n")
		if args.xtcfilename!="no":
			output_stat.write("\n")
			output_stat.write("4. nb frames processed:	" + str(nb_frames_processed) + " (" + str(nb_frames_xtc) + " frames in xtc, step=" + str(args.frames_dt) + ")\n")
		
		#what's in this file
		output_stat.write("\n")
		output_stat.write("Size range sampled by each lipid specie\n")
		output_stat.write("\n")
		output_stat.write("Note: the average below are not weighted by frames or number of lipds! this is a simple average on the unique cluster sizes detected throughout the xtc.\n")
		
		#species info for each leaflet
		for l in ["upper","lower"]:
			output_stat.write("\n")
			output_stat.write(str(l) + " leaflet\n")
			output_stat.write("=============\n")
			output_stat.write("specie	avg	min	max\n")
			output_stat.write("-----------------------------\n")
			for s in lipids_handled[l]:
				output_stat.write(str(s) + "	" + str(round(numpy.average(lipids_sizes_sampled[l][s]),1)) + "	" + str(numpy.min(lipids_sizes_sampled[l][s])) + "	" + str(numpy.max(lipids_sizes_sampled[l][s])) + "\n")
			output_stat.write("\n")
		output_stat.close()
		
		#groups
		#------
		if args.cluster_groups_file!="no":
			#create file
			if args.xtcfilename=="no":
				if args.m_algorithm=='connectivity':
					tmp_name=os.getcwd() + "/" + str(args.output_folder) + '/2_groups/2_0_' + args.grofilename[:-4] + '_annotated_clusterlip_' + str(args.m_algorithm) + '_c' + str(int(args.cutoff_connect)) + '_groups_sampled.stat'
				else:
					tmp_name=os.getcwd() + "/" + str(args.output_folder) + '/2_groups/2_0_' + args.grofilename[:-4] + '_annotated_clusterlip_' + str(args.m_algorithm) + '_r' + str(int(args.dbscan_dist)) + '_n' + str(args.dbscan_nb) + '_groups_sampled.stat'
			else:
				if args.m_algorithm=='connectivity':
					tmp_name=os.getcwd() + "/" + str(args.output_folder) + '/2_groups/2_0_' + args.xtcfilename[:-4] + '_annotated_clusterlip_' + str(args.m_algorithm) + '_c' + str(int(args.cutoff_connect)) + '_groups_sampled.stat'
				else:
					tmp_name=os.getcwd() + "/" + str(args.output_folder) + '/2_groups/2_0_' + args.xtcfilename[:-4] + '_annotated_clusterlip_' + str(args.m_algorithm) + '_r' + str(int(args.dbscan_dist)) + '_n' + str(args.dbscan_nb) + '_groups_sampled.stat'
			output_stat = open(tmp_name, 'w')
		
			#general info
			output_stat.write("[lipid clustering statistics - written by cluster_lip v" + str(version_nb) + "]\n")
			output_stat.write("\n")
			output_stat.write("1. Membrane composition:\n")
			output_stat.write(membrane_comp["upper"] + "\n")
			output_stat.write(membrane_comp["lower"] + "\n")
			tmp_string=str(lipids_handled["both"][0])
			for s in lipids_handled["both"][1:]:
				tmp_string+=", " + str(s)
			output_stat.write("\n")
			output_stat.write("2. Lipid species processed: " + str(tmp_string) + "\n")
			output_stat.write("\n")
			output_stat.write("3. Cluster detection Method:\n")
			if args.m_algorithm=='connectivity':
				output_stat.write(" - connectivity based\n")
				output_stat.write(" - contact cutoff = " + str(args.cutoff_connect) + " Angstrom\n")
			else:
				output_stat.write(" - density based (DBSCAN)\n")
				output_stat.write(" - search radius = " + str(args.dbscan_dist) + " Angstrom, nb of neighbours = " + str(args.dbscan_nb) + "\n")
			
			#what's in this file
			output_stat.write("\n")
			output_stat.write("Group index range sampled by each lipid specie\n")
			output_stat.write("group sizes:\n")
			for g_index in range(0,groups_nb):
				if groups_boundaries[g_index][1]==100000:
					output_stat.write(str(g_index) + "=" + str(groups_boundaries[g_index][0]) + "+\n")
				else:
					output_stat.write(str(g_index) + "=" + str(groups_boundaries[g_index][0]) + "-" + str(groups_boundaries[g_index][1]) + "\n")
		
			#species info for each leaflet
			for l in ["upper","lower"]:
				output_stat.write("\n")
				output_stat.write(str(l) + " leaflet\n")
				output_stat.write("=============\n")
				output_stat.write("specie	min	max\n")
				output_stat.write("--------------------\n")
				for s in lipids_handled[l]:
					output_stat.write(str(s) + "	" + str(numpy.min(lipids_groups_sampled[l][s])) + "	" + str(numpy.max(lipids_groups_sampled[l][s])) + "\n")
				output_stat.write("\n")
			output_stat.close()		

	#case: xtc snapshot
	#==================
	else:
		#sizes
		#-----
		#create file
		if args.m_algorithm=='connectivity':
			tmp_name=os.getcwd() + '/' + str(args.output_folder) + '/3_snapshots/sizes/' + args.xtcfilename[:-4] + '_annotated_clusterlip_' + str(args.m_algorithm) + '_c' + str(int(args.cutoff_connect)) + '_sizes_' + str(int(t)).zfill(5) + 'ns.stat'
		else:
			tmp_name=os.getcwd() + '/' + str(args.output_folder) + '/3_snapshots/sizes/' + args.xtcfilename[:-4] + '_annotated_clusterlip_' + str(args.m_algorithm) + '_r' + str(int(args.dbscan_dist)) + '_n' + str(args.dbscan_nb) + '_sizes_' + str(int(t)).zfill(5) + 'ns.stat'
		output_stat = open(tmp_name, 'w')
		
		#general info
		output_stat.write("[lipid clustering statistics - written by cluster_lip v" + str(version_nb) + "]\n")
		output_stat.write("\n")
		output_stat.write("1. membrane composition:\n")
		output_stat.write(membrane_comp["upper"] + "\n")
		output_stat.write(membrane_comp["lower"] + "\n")
		tmp_string=str(lipids_handled["both"][0])
		for s in lipids_handled["both"][1:]:
			tmp_string+=", " + str(s)
		output_stat.write("\n")
		output_stat.write("2. lipid species processed: " + str(tmp_string) + "\n")
		output_stat.write("\n")
		output_stat.write("3. cluster detection Method:\n")
		if args.m_algorithm=='connectivity':
			output_stat.write(" - connectivity based\n")
			output_stat.write(" - contact cutoff = " + str(args.cutoff_connect) + " Angstrom\n")
		else:
			output_stat.write(" - density based (DBSCAN)\n")
			output_stat.write(" - search radius = " + str(args.dbscan_dist) + " Angstrom, nb of neighbours = " + str(args.dbscan_nb) + "\n")
		output_stat.write("\n")
		output_stat.write("4. time: " + str(t) + "ns (frame " + str(f_nb) + "/" + str(nb_frames_xtc) + ")\n")
		
		#what's in this file
		output_stat.write("\n")
		output_stat.write("Size range sampled by each lipid specie (avg = avg size of the cluster a lipid is involved in)\n")
	
		#species info for each leaflet
		for l in ["upper","lower"]:
			output_stat.write("\n")
			output_stat.write(str(l) + " leaflet\n")
			output_stat.write("=============\n")
			output_stat.write("specie	avg	min	max\n")
			output_stat.write("-----------------------------\n")
			for s in lipids_handled[l]:
				output_stat.write(str(s) + "	" + str(round(numpy.average(lipids_cluster_size_mat[l][s][:,f_index]),1)) + "	" + str(numpy.min(lipids_cluster_size_mat[l][s][:,f_index])) + "	" + str(numpy.max(lipids_cluster_size_mat[l][s][:,f_index])) + "\n")
			output_stat.write("\n")
		output_stat.close()

		#groups
		#------
		if args.cluster_groups_file!="no":
			#create file
			if args.m_algorithm=='connectivity':
				tmp_name=os.getcwd() + '/' + str(args.output_folder) + '/3_snapshots/groups/' + args.xtcfilename[:-4] + '_annotated_clusterlip_' + str(args.m_algorithm) + '_c' + str(int(args.cutoff_connect)) + '_groups_' + str(int(t)).zfill(5) + 'ns.stat'
			else:
				tmp_name=os.getcwd() + '/' + str(args.output_folder) + '/3_snapshots/groups/' + args.xtcfilename[:-4] + '_annotated_clusterlip_' + str(args.m_algorithm) + '_r' + str(int(args.dbscan_dist)) + '_n' + str(args.dbscan_nb) + '_groups_' + str(int(t)).zfill(5) + 'ns.txt'
			output_stat = open(tmp_name, 'w')

			#general info
			output_stat.write("[lipid clustering statistics - written by cluster_lip v" + str(version_nb) + "]\n")
			output_stat.write("\n")
			output_stat.write("1. Membrane composition:\n")
			output_stat.write(membrane_comp["upper"] + "\n")
			output_stat.write(membrane_comp["lower"] + "\n")
			tmp_string=str(lipids_handled["both"][0])
			for s in lipids_handled["both"][1:]:
				tmp_string+=", " + str(s)
			output_stat.write("\n")
			output_stat.write("2. Lipid species processed: " + str(tmp_string) + "\n")
			output_stat.write("\n")
			output_stat.write("3. Cluster detection Method:\n")
			if args.m_algorithm=='connectivity':
				output_stat.write(" - connectivity based\n")
				output_stat.write(" - contact cutoff = " + str(args.cutoff_connect) + " Angstrom\n")
			else:
				output_stat.write(" - density based (DBSCAN)\n")
				output_stat.write(" - search radius = " + str(args.dbscan_dist) + " Angstrom, nb of neighbours = " + str(args.dbscan_nb) + "\n")
			output_stat.write("\n")
			output_stat.write("4. Time: " + str(t) + "ns (frame " + str(f_nb) + "/" + str(nb_frames_xtc) + ")\n")
			
			#what's in this file
			output_stat.write("\n")
			output_stat.write("Group index range sampled by each lipid specie\n")
			output_stat.write("group sizes:\n")
			for g_index in range(0,groups_nb):
				if groups_boundaries[g_index][1]==100000:
					output_stat.write(str(g_index) + "=" + str(groups_boundaries[g_index][0]) + "+\n")
				else:
					output_stat.write(str(g_index) + "=" + str(groups_boundaries[g_index][0]) + "-" + str(groups_boundaries[g_index][1]) + "\n")
		
			#species info for each leaflet
			for l in ["upper","lower"]:
				output_stat.write("\n")
				output_stat.write(str(l) + " leaflet\n")
				output_stat.write("=============\n")
				output_stat.write("specie	min	max\n")
				output_stat.write("--------------------\n")
				for s in lipids_handled[l]:
					output_stat.write(str(s) + "	" + str(numpy.min(lipids_cluster_group_mat[l][s][:,f_index])) + "	" + str(numpy.max(lipids_cluster_group_mat[l][s][:,f_index])) + "\n")
				output_stat.write("\n")
			output_stat.close()
	
	return
def write_frame_snapshot(f_index, t):
	
	#sizes
	#=====
	#store cluster info in beta factor field
	for l in ["lower","upper"]:
		for s in lipids_handled[l]:
			for r_index in range(0,lipids_sele_nb[l][s]):
				lipids_selection[l][s][r_index].set_bfactor(lipids_cluster_size_mat[l][s][r_index,f_index])
			
	#write annotated file
	if args.xtcfilename=="no":
		if args.m_algorithm=='connectivity':
			all_atoms.write(os.getcwd() + '/' + str(args.output_folder) + '/3_snapshots/sizes/' + args.grofilename[:-4] + '_annotated_clusterlip_' + str(args.m_algorithm) + '_c' + str(int(args.cutoff_connect)) + '_sizes', format="PDB")
		else:
			all_atoms.write(os.getcwd() + '/' + str(args.output_folder) + '/3_snapshots/sizes/' + args.grofilename[:-4] + '_annotated_clusterlip_' + str(args.m_algorithm) + '_r' + str(int(args.dbscan_dist)) + '_n' + str(args.dbscan_nb) + '_sizes', format="PDB")
	else:
		if args.m_algorithm=='connectivity':
			tmp_name=os.getcwd() + "/" + str(args.output_folder) + '/3_snapshots/sizes/' + args.xtcfilename[:-4] + '_annotated_clusterlip_' + str(args.m_algorithm) + '_c' + str(int(args.cutoff_connect)) + '_sizes_' + str(int(t)).zfill(5) + 'ns.pdb'
		else:
			tmp_name=os.getcwd() + "/" + str(args.output_folder) + '/3_snapshots/sizes/' + args.xtcfilename[:-4] + '_annotated_clusterlip_' + str(args.m_algorithm) + '_r' + str(int(args.dbscan_dist)) + '_n' + str(args.dbscan_nb) + '_sizes_' + str(int(t)).zfill(5) + 'ns.pdb'
		W=Writer(tmp_name, nb_atoms)
		W.write(all_atoms)
	
	#groups
	#======
	if args.cluster_groups_file!="no":
		#store cluster info in beta factor field
		for l in ["lower","upper"]:
			for s in lipids_handled[l]:
				for r_index in range(0,lipids_sele_nb[l][s]):
					lipids_selection[l][s][r_index].set_bfactor(lipids_cluster_group_mat[l][s][r_index,f_index])
		
		#write annotated file
		if args.xtcfilename=="no":
			if args.m_algorithm=='connectivity':
				all_atoms.write(os.getcwd() + '/' + str(args.output_folder) + '/3_snapshots/groups/' + args.grofilename[:-4] + '_annotated_clusterlip_' + str(args.m_algorithm) + '_c' + str(int(args.cutoff_connect)) + '_groups', format="PDB")
			else:
				all_atoms.write(os.getcwd() + '/' + str(args.output_folder) + '/3_snapshots/groups/' + args.grofilename[:-4] + '_annotated_clusterlip_' + str(args.m_algorithm) + '_r' + str(int(args.dbscan_dist)) + '_n' + str(args.dbscan_nb) + '_groups', format="PDB")
		else:
			if args.m_algorithm=='connectivity':
				tmp_name=os.getcwd() + "/" + str(args.output_folder) + '/3_snapshots/groups/' + args.xtcfilename[:-4] + '_annotated_clusterlip_' + str(args.m_algorithm) + '_c' + str(int(args.cutoff_connect)) + '_groups_' + str(int(t)).zfill(5) + 'ns.pdb'
			else:
				tmp_name=os.getcwd() + "/" + str(args.output_folder) + '/3_snapshots/groups/' + args.xtcfilename[:-4] + '_annotated_clusterlip_' + str(args.m_algorithm) + '_r' + str(int(args.dbscan_dist)) + '_n' + str(args.dbscan_nb) + '_groups_' + str(int(t)).zfill(5) + 'ns.pdb'
			W=Writer(tmp_name, nb_atoms)
			W.write(all_atoms)
		
	return
def write_frame_annotation(f_index,t):
	
	#sizes
	#=====
	#create file
	if args.xtcfilename=="no":
		if args.m_algorithm=='connectivity':
			filename_details=os.getcwd() + '/' + str(args.output_folder) + '/3_snapshots/sizes/' + args.grofilename[:-4] + '_annotated_clusterlip_' + str(args.m_algorithm) + '_c' + str(int(args.cutoff_connect)) + '_sizes_.txt'
		else:
			filename_details=os.getcwd() + '/' + str(args.output_folder) + '/3_snapshots/sizes/' + args.grofilename[:-4] + '_annotated_clusterlip_' + str(args.m_algorithm) + '_r' + str(int(args.dbscan_dist)) + '_n' + str(args.dbscan_nb) + '_sizes.txt'
	else:
		if args.m_algorithm=='connectivity':
			filename_details=os.getcwd() + '/' + str(args.output_folder) + '/3_snapshots/sizes/' + args.xtcfilename[:-4] + '_annotated_clusterlip_' + str(args.m_algorithm) + '_c' + str(int(args.cutoff_connect)) + '_sizes_' + str(int(t)).zfill(5) + 'ns.txt'
		else:
			filename_details=os.getcwd() + '/' + str(args.output_folder) + '/3_snapshots/sizes/' + args.xtcfilename[:-4] + '_annotated_clusterlip_' + str(args.m_algorithm) + '_r' + str(int(args.dbscan_dist)) + '_n' + str(args.dbscan_nb) + '_sizes_' + str(int(t)).zfill(5) + 'ns.txt'
	output_stat = open(filename_details, 'w')		
	
	#create selection string
	tmp_sele_string=""
	for l in ["lower","upper"]:
		for s in lipids_handled[l]:
			for r_index in range(0,lipids_sele_nb[l][s]):
				tmp_sele_string+="." + lipids_selection_VMD_string[l][s][r_index]
	tmp_sele_string=tmp_sele_string[1:]
	output_stat.write(tmp_sele_string + "\n")

	#write min and max boundaries of thickness
	output_stat.write(str(numpy.min(lipids_sizes_sampled[l][s])) + "." + str(numpy.max(lipids_sizes_sampled[l][s])) + "\n")
	
	#ouptut cluster info for each lipid
	tmp_clustlip="1"
	for l in ["lower","upper"]:
		for s in lipids_handled[l]:
			for r_index in range(0,lipids_sele_nb[l][s]):
				tmp_clustlip+=";" + str(lipids_cluster_size_mat[l][s][r_index,f_index])
	output_stat.write(tmp_clustlip + "\n")
	output_stat.close()

	#groups
	#======
	if args.cluster_groups_file!="no":
		if args.xtcfilename=="no":
			if args.m_algorithm=='connectivity':
				filename_details=os.getcwd() + '/' + str(args.output_folder) + '/3_snapshots/groups/' + args.grofilename[:-4] + '_annotated_clusterlip_' + str(args.m_algorithm) + '_c' + str(int(args.cutoff_connect)) + '_groups_.txt'
			else:
				filename_details=os.getcwd() + '/' + str(args.output_folder) + '/3_snapshots/groups/' + args.grofilename[:-4] + '_annotated_clusterlip_' + str(args.m_algorithm) + '_r' + str(int(args.dbscan_dist)) + '_n' + str(args.dbscan_nb) + '_groups.txt'
		else:
			if args.m_algorithm=='connectivity':
				filename_details=os.getcwd() + '/' + str(args.output_folder) + '/3_snapshots/groups/' + args.xtcfilename[:-4] + '_annotated_clusterlip_' + str(args.m_algorithm) + '_c' + str(int(args.cutoff_connect)) + '_groups_' + str(int(t)).zfill(5) + 'ns.txt'
			else:
				filename_details=os.getcwd() + '/' + str(args.output_folder) + '/3_snapshots/groups/' + args.xtcfilename[:-4] + '_annotated_clusterlip_' + str(args.m_algorithm) + '_r' + str(int(args.dbscan_dist)) + '_n' + str(args.dbscan_nb) + '_groups_' + str(int(t)).zfill(5) + 'ns.txt'
		output_stat = open(filename_details, 'w')		
		
		#create selection string
		tmp_sele_string=""
		for l in ["lower","upper"]:
			for s in lipids_handled[l]:
				for r_index in range(0,lipids_sele_nb[l][s]):
					tmp_sele_string+="." + lipids_selection_VMD_string[l][s][r_index]
		output_stat.write(tmp_sele_string[1:] + "\n")
	
		#write min and max boundaries of thickness
		output_stat.write(str(numpy.min(lipids_groups_sampled[l][s])) + "." + str(numpy.max(lipids_groups_sampled[l][s])) + "\n")
		
		#ouptut cluster info for each lipid
		tmp_clustlip="1"
		for l in ["lower","upper"]:
			for s in lipids_handled[l]:
				for r_index in range(0,lipids_sele_nb[l][s]):
					tmp_clustlip+=";" + str(lipids_cluster_group_mat[l][s][r_index,f_index])
		output_stat.write(tmp_clustlip + "\n")
		output_stat.close()
	
	return
def write_xtc_snapshots():
	
	#NB: - this will always output the first and final frame snapshots
	#    - it will also intermediate frames according to the -w option
	
	loc_nb_frames_processed=0
	for ts in U.trajectory:

		#case: frames before specified time boundaries
		#---------------------------------------------
		if ts.time/float(1000)<args.t_start:
			progress='\r -skipping frame ' + str(ts.frame) + '/' + str(nb_frames_xtc) + '        '
			sys.stdout.flush()
			sys.stdout.write(progress)

		#case: frames within specified time boundaries
		#---------------------------------------------
		elif ts.time/float(1000)>args.t_start and ts.time/float(1000)<args.t_end:
			progress='\r -writing snapshots...   frame ' + str(ts.frame) + '/' + str(nb_frames_xtc) + '        '
			sys.stdout.flush()
			sys.stdout.write(progress)
			if ((ts.frame-1) % args.frames_dt)==0:
				if ((loc_nb_frames_processed) % args.frames_write_dt)==0 or loc_nb_frames_processed==nb_frames_processed-1:
					write_frame_stat(ts.frame, loc_nb_frames_processed, ts.time/float(1000))
					write_frame_snapshot(loc_nb_frames_processed, ts.time/float(1000))
					write_frame_annotation(loc_nb_frames_processed, ts.time/float(1000))
				loc_nb_frames_processed+=1
		
		#case: frames after specified time boundaries
		#---------------------------------------------
		elif ts.time/float(1000)>args.t_end:
			break

	print ''

	return
def write_xtc_annotation():
	
	#sizes
	#=====
	#create file
	if args.m_algorithm=='connectivity':
		filename_details=os.getcwd() + '/' + str(args.output_folder) + '/4_VMD/' + args.xtcfilename[:-4] + '_annotated_clusterlip_dt' + str(args.frames_dt) + '_' + str(args.m_algorithm) + '_c' + str(int(args.cutoff_connect)) + '_sizes.txt'
	else:
		filename_details=os.getcwd() + '/' + str(args.output_folder) + '/4_VMD/' + args.xtcfilename[:-4] + '_annotated_clusterlip_dt' + str(args.frames_dt) + '_' + str(args.m_algorithm) + '_r' + str(int(args.dbscan_dist)) + '_n' + str(args.dbscan_nb) + '_sizes.txt'
	output_stat = open(filename_details, 'w')		
	
	#create selection string
	tmp_sele_string=""
	for l in ["lower","upper"]:
		for s in lipids_handled[l]:
			for r_index in range(0,lipids_sele_nb[l][s]):
				tmp_sele_string+="." + lipids_selection_VMD_string[l][s][r_index]
	tmp_sele_string=tmp_sele_string[1:]
	output_stat.write(tmp_sele_string + "\n")

	#write min and max boundaries of thickness
	output_stat.write(str(numpy.min(lipids_sizes_sampled[l][s])) + "." + str(numpy.max(lipids_sizes_sampled[l][s])) + "\n")
	
	#ouptut cluster info for each lipid
	for frame in sorted(time_stamp.iterkeys()):
		tmp_clustlip=str(frame)
		frame_index=sorted(time_stamp.keys()).index(frame)
		for l in ["lower","upper"]:
			for s in lipids_handled[l]:
				for r_index in range(0,lipids_sele_nb[l][s]):
					tmp_clustlip+=";" + str(lipids_cluster_size_mat[l][s][r_index,frame_index])
		output_stat.write(tmp_clustlip + "\n")
	output_stat.close()

	#groups
	#======
	if args.cluster_groups_file!="no":
		#create file
		if args.m_algorithm=='connectivity':
			filename_details=os.getcwd() + '/' + str(args.output_folder) + '/4_VMD/' + args.xtcfilename[:-4] + '_annotated_clusterlip_dt' + str(args.frames_dt) + '_' + str(args.m_algorithm) + '_c' + str(int(args.cutoff_connect)) + '_groups.txt'
		else:
			filename_details=os.getcwd() + '/' + str(args.output_folder) + '/4_VMD/' + args.xtcfilename[:-4] + '_annotated_clusterlip_dt' + str(args.frames_dt) + '_' + str(args.m_algorithm) + '_r' + str(int(args.dbscan_dist)) + '_n' + str(args.dbscan_nb) + '_groups.txt'
		output_stat = open(filename_details, 'w')		
		
		#create selection string
		tmp_sele_string=""
		for l in ["lower","upper"]:
			for s in lipids_handled[l]:
				for r_index in range(0,lipids_sele_nb[l][s]):
					tmp_sele_string+="." + lipids_selection_VMD_string[l][s][r_index]
		tmp_sele_string=tmp_sele_string[1:]
		output_stat.write(tmp_sele_string + "\n")
	
		#write min and max boundaries of thickness
		output_stat.write(str(numpy.min(lipids_groups_sampled[l][s])) + "." + str(numpy.max(lipids_groups_sampled[l][s])) + "\n")
		
		#ouptut cluster info for each lipid
		for frame in sorted(time_stamp.iterkeys()):
			tmp_clustlip=str(frame)
			frame_index=sorted(time_stamp.keys()).index(frame)
			for l in ["lower","upper"]:
				for s in lipids_handled[l]:
					for r_index in range(0,lipids_sele_nb[l][s]):
						tmp_clustlip+=";" + str(lipids_cluster_group_mat[l][s][r_index,frame_index])
			output_stat.write(tmp_clustlip + "\n")
		output_stat.close()
	
	return

################################################################################################################################################
# DATA STRUCTURES
################################################################################################################################################

#time
time_stamp={}

#cluster size/size group each lipid is involved in at each frame
lipids_cluster_size={}
lipids_cluster_group={}
lipids_cluster_size_mat={}
lipids_cluster_group_mat={}
for l in ["lower","upper"]:
	lipids_cluster_size[l]={}
	lipids_cluster_group[l]={}
	lipids_cluster_size_mat[l]={}
	lipids_cluster_group_mat[l]={}
	for s in lipids_handled[l]:
		lipids_cluster_size[l][s]={}
		lipids_cluster_group[l][s]={}
		for r_index in range(0,lipids_sele_nb[l][s]):
			lipids_cluster_size[l][s][r_index]=[]
			lipids_cluster_group[l][s][r_index]=[]

sizes_colors_nb={}
sizes_colors_dict={}
sizes_colors_list={}
for s in lipids_handled["both"]:
	sizes_colors_nb[s]=0
	sizes_colors_list[s]=[]
	sizes_colors_dict[s]={}

#sizes sampled by each specie
lipids_sizes_sampled={}
lipids_groups_sampled={}
for l in ["lower","upper","both"]:
	lipids_sizes_sampled[l]={}
	lipids_groups_sampled[l]={}
	for s in lipids_handled[l]:
		lipids_sizes_sampled[l][s]=[]
		lipids_groups_sampled[l][s]=[]

sizes_pc={}
groups_pc={}
groups_stability={}
for l in ["lower","upper"]:
	sizes_pc[l]={}
	groups_pc[l]={}
	groups_stability[l]={}
	for s in lipids_handled[l]:
		sizes_pc[l][s]={}
		groups_pc[l][s]={}
		groups_stability[l][s]={}

#smooth data
if args.nb_smoothing>1:
	time_sorted=[]
	time_smoothed=[]
	groups_pc_smoothed={}
	for l in ["lower","upper"]:
		groups_pc_smoothed[l]={}
		for s in lipids_handled[l]:
			groups_pc_smoothed[l][s]={}

################################################################################################################################################
# ALGORITHM : Browse trajectory and process relevant frames
################################################################################################################################################

print "\nDetecting lipid clusters..."

#case: gro file
#==============
if args.xtcfilename=="no":
	#store dummy time
	time_stamp[1]=0
	
	#browse each specie in each leaflet
	for l in ["lower","upper"]:
		print " -" + str(l) + " leaflet..."
		for s in lipids_handled[l]:
			#detect clusters
			if args.m_algorithm=='connectivity':
				tmp_groups=detect_clusters_connectivity(lipids_sele[l][s].coordinates(), U.dimensions)
			elif args.m_algorithm=='density':
				tmp_groups=clusters=detect_clusters_density(lipids_sele[l][s].coordinates(), U.dimensions)
			#case: store cluster size only for each lipid
			if args.cluster_groups_file=="no":
				for g in tmp_groups:
					for r_index in g:
						lipids_cluster_size[l][s][r_index].append(numpy.size(g))
			#case: store cluster size and group size for each lipid
			else:
				for g in tmp_groups:
					for r_index in g:
						lipids_cluster_size[l][s][r_index].append(numpy.size(g))
						lipids_cluster_group[l][s][r_index].append(groups_sizes_dict[numpy.size(g)])			

#case: xtc file
#==============
else:
	for ts in U.trajectory:

		#case: frames before specified time boundaries
		#---------------------------------------------
		if ts.time/float(1000)<args.t_start:
			progress='\r -skipping frame ' + str(ts.frame) + '/' + str(nb_frames_xtc) + '        '
			sys.stdout.flush()
			sys.stdout.write(progress)

		#case: frames within specified time boundaries
		#---------------------------------------------
		elif ts.time/float(1000)>args.t_start and ts.time/float(1000)<args.t_end:
			progress='\r -processing frame ' + str(ts.frame) + '/' + str(nb_frames_xtc) + '        '
			sys.stdout.flush()
			sys.stdout.write(progress)
			if ((ts.frame-1) % args.frames_dt)==0:
				nb_frames_processed+=1
				#store time
				time_stamp[ts.frame]=ts.time/float(1000)

				#browse each specie in each leaflet
				for l in ["lower","upper"]:
					for s in lipids_handled[l]:
						#detect clusters
						if args.m_algorithm=='connectivity':
							tmp_groups=detect_clusters_connectivity(lipids_sele[l][s].coordinates(), U.dimensions)
						elif args.m_algorithm=='density':
							tmp_groups=detect_clusters_density(lipids_sele[l][s].coordinates(), U.dimensions)
						#case: store cluster size only for each lipd
						if args.cluster_groups_file=="no":
							for g in tmp_groups:
								for r_index in g:
									lipids_cluster_size[l][s][r_index].append(numpy.size(g))
						#case: store cluster size and group size for each lipd
						else:
							for g in tmp_groups:
								for r_index in g:
									lipids_cluster_size[l][s][r_index].append(numpy.size(g))
									lipids_cluster_group[l][s][r_index].append(groups_sizes_dict[numpy.size(g)]) 		#handle case of "other"...
								
		#case: frames after specified time boundaries
		#--------------------------------------------
		elif ts.time/float(1000)>args.t_end:
			break
									
	print ""

################################################################################################################################################
# CALCULATE STATISTICS
################################################################################################################################################

print "\nCalculating statistics..."
get_sizes_sampled()
update_color_dict()
calc_stat()
if args.xtcfilename!="no" and args.cluster_groups_file!="no":
	smooth_data()

################################################################################################################################################
# PRODUCE OUTPUTS
################################################################################################################################################

print "\nWriting outputs..."

#case: gro file
if args.xtcfilename=="no":
	print " -writing statistics..."
	write_frame_stat(1, "all","all")
	print " -writing annotated pdb..."
	write_frame_snapshot(0,0)
	write_frame_annotation(0,0)

#case: xtc file
else:
	if len(lipids_sizes_sampled["both"]["all"])>1:
		#writing statistics
		print " -writing statistics..."
		write_frame_stat(0, "all", "all")
		#output cluster snapshots
		write_xtc_snapshots()
		#write annotation files for VMD
		print " -writing VMD annotation files..."
		write_xtc_annotation()
		#write xvg and graphs
		print " -writing xvg and graphs..."
		graph_aggregation_2D_sizes()
		if args.cluster_groups_file!="no":
			graph_aggregation_2D_groups()
			write_stability_groups()
			write_xvg_groups()
			graph_xvg_groups()
			if args.nb_smoothing>1:
				write_xvg_groups_smoothed()
				graph_xvg_groups_smoothed()
	else:
		print "\n"
		print "Warning: a single cluster size (", str(lipids_sizes_sampled["both"]["all"][0]), ") was detected throughout the trajectory. Check the -m, -c, -r or -n options (see cluster_prot -h)."
		write_warning()
	
#exit
#====
print "\nFinished successfully! Check output in ./" + args.output_folder + "/"
print ""
sys.exit(0)
