//genesis
// kkit Version 11 flat dumpfile

// Saved on 100
include kkit {argv 1}
FASTDT = 0.001
SIMDT = 0.001
CONTROLDT = 0.1
PLOTDT = 0.1
MAXTIME = 100
TRANSIENT_TIME = 2
VARIABLE_DT_FLAG = 0
DEFAULT_VOL = 1.7291666666666674e-21
VERSION = 11.0 
setfield /file/modpath value ~/scripts/modules
kparms

//genesis
initdump -version 3 -ignoreorphans 1
simobjdump table input output alloced step_mode stepsize x y z
simobjdump xtree path script namemode sizescale
simobjdump xcoredraw xmin xmax ymin ymax
simobjdump xtext editable
simobjdump xgraph xmin xmax ymin ymax overlay
simobjdump xplot pixflags script fg ysquish do_slope wy
simobjdump group xtree_fg_req xtree_textfg_req plotfield expanded movealone \
  link savename file version md5sum mod_save_flag x y z
simobjdump geometry size dim shape outside xtree_fg_req xtree_textfg_req x y z
simobjdump kpool DiffConst CoInit Co n nInit mwt nMin vol slave_enable \
  geomname xtree_fg_req xtree_textfg_req x y z
simobjdump kreac kf kb notes xtree_fg_req xtree_textfg_req x y z
simobjdump kenz CoComplexInit CoComplex nComplexInit nComplex vol k1 k2 k3 \
  keepconc usecomplex notes xtree_fg_req xtree_textfg_req link x y z
simobjdump stim level1 width1 delay1 level2 width2 delay2 baselevel trig_time \
  trig_mode notes xtree_fg_req xtree_textfg_req is_running x y z
simobjdump xtab input output alloced step_mode stepsize notes editfunc \
  xtree_fg_req xtree_textfg_req baselevel last_x last_y is_running x y z
simobjdump kchan perm gmax Vm is_active use_nernst notes xtree_fg_req \
  xtree_textfg_req x y z
simobjdump transport input output alloced step_mode stepsize dt delay clock \
  kf xtree_fg_req xtree_textfg_req x y z
simobjdump proto x y z
simundump geometry /kinetics/geometry 0 1.7291666666666674e-21 3 sphere  "" white black 9 5 0
simundump group /kinetics/GABA 0 blue green x 0 0 "" defaultfile \
  defaultfile.g 0 0 0 6 5 0
simundump kpool /kinetics/GABA/Ca 0 0.0 0 0 0 0.08330930182075003 0 0 1.0413286343750006 0 /kinetics/geometry 62 yellow 7 3 0
simundump kpool /kinetics/GABA/Ca_ext 0 0.0 0 0 0 0.08330930182075003 0 0 1.0413286343750006 4 /kinetics/geometry 52 yellow 11 3 0
simundump kpool /kinetics/GABA/RR_pool 0 0.0 0 0 0 0.2603271401425001 0 0 1.0413286343750006 0 /kinetics/geometry 1 yellow 5 1 0
simundump kpool /kinetics/GABA/vesicle_pool 0 0.0 0 0 0 0.3219757673177498 0 0 1.0413286343750006 4 /kinetics/geometry 27 yellow 11 -1 0
simundump kpool /kinetics/GABA/Docked 0 0.0 0 0 0 0.0 0 0 1.0413286343750006 0 /kinetics/geometry 54 blue 9 1 0
simundump kpool /kinetics/GABA/Ca_RR 0 0.0 0 0 0 0.0 0 0 1.0413286343750006 0 /kinetics/geometry 51 blue 7 1 0
simundump kpool /kinetics/GABA/Receptor 0 0.0 0 0 0 15.166763367750006 0 0 1.0413286343750006 0 /kinetics/geometry 12 yellow 13 3 0
simundump kpool /kinetics/GABA/L_R 0 0.0 0 0 0 0.0 0 0 1.0413286343750006 0 /kinetics/geometry 23 yellow 13 1 0
simundump kpool /kinetics/GABA/GABA 0 0.0 0 0 0 0.0 0 0 1.0413286343750006 0 /kinetics/geometry 7 yellow 11 1 0
simundump kreac /kinetics/GABA/remove_Ca 0 14.404842529854205 14.404842529854205 "" white yellow 9 4 0
simundump kreac /kinetics/GABA/remove 0 71417.78413359054 0.0 "" white yellow 9 0 0
simundump kreac /kinetics/GABA/replenish_vesicle 0 3.174145444102718 3.174145444102718 "" white yellow 9 -2 0
simundump kreac /kinetics/GABA/vesicle_release 0 21.737752943824336 0.0 "" white yellow 10 2 0
simundump kreac /kinetics/GABA/Ca_bind_RR 0 62.693318751999215 38.15684659129625 "" white blue 6 2 0
simundump kreac /kinetics/GABA/docking 0 1162.064875818678 0.0 "" white blue 8 2 0
simundump kreac /kinetics/GABA/ligand_binding 0 291.8209641910736 9.43444024775322 "" white blue 12 2 0
simundump kreac /kinetics/GABA/undocking 0 5.0 0.0 "" white blue 7 0 0
simundump xgraph /graphs/conc1 0 0 99 0.001 0.999 0
simundump xgraph /graphs/conc2 0 0 100 0 1 0
 simundump xplot /graphs/conc1/Docked.Co 3 524288 \
"delete_plot.w <s> <d>; edit_plot.D <w>" blue 0 0 1
simundump xgraph /moregraphs/conc3 0 0 100 0 1 0
simundump xgraph /moregraphs/conc4 0 0 100 0 1 0
 simundump xcoredraw /edit/draw 0 -6 4 -2 6
simundump xtree /edit/draw/tree 0 \
  /kinetics/#[],/kinetics/#[]/#[],/kinetics/#[]/#[]/#[][TYPE!=proto],/kinetics/#[]/#[]/#[][TYPE!=linkinfo]/##[] "edit_elm.D <v>; drag_from_edit.w <d> <S> <x> <y> <z>" auto 0.6
simundump xtext /file/notes 0 1
addmsg /kinetics/GABA/Ca /kinetics/GABA/remove_Ca SUBSTRATE n 
addmsg /kinetics/GABA/remove_Ca /kinetics/GABA/Ca REAC A B 
addmsg /kinetics/GABA/Ca /kinetics/GABA/remove_Ca SUBSTRATE n 
addmsg /kinetics/GABA/remove_Ca /kinetics/GABA/Ca REAC A B 
addmsg /kinetics/GABA/Ca_ext /kinetics/GABA/remove_Ca PRODUCT n 
addmsg /kinetics/GABA/remove_Ca /kinetics/GABA/Ca_ext REAC B A
addmsg /kinetics/GABA/Ca_ext /kinetics/GABA/remove_Ca PRODUCT n 
addmsg /kinetics/GABA/remove_Ca /kinetics/GABA/Ca_ext REAC B A
addmsg /kinetics/GABA/GABA /kinetics/GABA/remove SUBSTRATE n 
addmsg /kinetics/GABA/remove /kinetics/GABA/GABA REAC A B 
addmsg /kinetics/GABA/Ca /kinetics/GABA/remove PRODUCT n 
addmsg /kinetics/GABA/remove /kinetics/GABA/Ca REAC B A
addmsg /kinetics/GABA/Ca /kinetics/GABA/remove PRODUCT n 
addmsg /kinetics/GABA/remove /kinetics/GABA/Ca REAC B A
addmsg /kinetics/GABA/vesicle_pool /kinetics/GABA/remove PRODUCT n 
addmsg /kinetics/GABA/remove /kinetics/GABA/vesicle_pool REAC B A
addmsg /kinetics/GABA/vesicle_pool /kinetics/GABA/replenish_vesicle SUBSTRATE n 
addmsg /kinetics/GABA/replenish_vesicle /kinetics/GABA/vesicle_pool REAC A B 
addmsg /kinetics/GABA/RR_pool /kinetics/GABA/replenish_vesicle PRODUCT n 
addmsg /kinetics/GABA/replenish_vesicle /kinetics/GABA/RR_pool REAC B A
addmsg /kinetics/GABA/Ca /kinetics/GABA/vesicle_release SUBSTRATE n 
addmsg /kinetics/GABA/vesicle_release /kinetics/GABA/Ca REAC A B 
addmsg /kinetics/GABA/Ca /kinetics/GABA/vesicle_release SUBSTRATE n 
addmsg /kinetics/GABA/vesicle_release /kinetics/GABA/Ca REAC A B 
addmsg /kinetics/GABA/Docked /kinetics/GABA/vesicle_release SUBSTRATE n 
addmsg /kinetics/GABA/vesicle_release /kinetics/GABA/Docked REAC A B 
addmsg /kinetics/GABA/GABA /kinetics/GABA/vesicle_release PRODUCT n 
addmsg /kinetics/GABA/vesicle_release /kinetics/GABA/GABA REAC B A
addmsg /kinetics/GABA/Ca /kinetics/GABA/Ca_bind_RR SUBSTRATE n 
addmsg /kinetics/GABA/Ca_bind_RR /kinetics/GABA/Ca REAC A B 
addmsg /kinetics/GABA/Ca /kinetics/GABA/Ca_bind_RR SUBSTRATE n 
addmsg /kinetics/GABA/Ca_bind_RR /kinetics/GABA/Ca REAC A B 
addmsg /kinetics/GABA/RR_pool /kinetics/GABA/Ca_bind_RR SUBSTRATE n 
addmsg /kinetics/GABA/Ca_bind_RR /kinetics/GABA/RR_pool REAC A B 
addmsg /kinetics/GABA/Ca_RR /kinetics/GABA/Ca_bind_RR PRODUCT n 
addmsg /kinetics/GABA/Ca_bind_RR /kinetics/GABA/Ca_RR REAC B A
addmsg /kinetics/GABA/Ca_RR /kinetics/GABA/docking SUBSTRATE n 
addmsg /kinetics/GABA/docking /kinetics/GABA/Ca_RR REAC A B 
addmsg /kinetics/GABA/Ca /kinetics/GABA/docking PRODUCT n 
addmsg /kinetics/GABA/docking /kinetics/GABA/Ca REAC B A
addmsg /kinetics/GABA/Ca /kinetics/GABA/docking PRODUCT n 
addmsg /kinetics/GABA/docking /kinetics/GABA/Ca REAC B A
addmsg /kinetics/GABA/Docked /kinetics/GABA/docking PRODUCT n 
addmsg /kinetics/GABA/docking /kinetics/GABA/Docked REAC B A
addmsg /kinetics/GABA/Receptor /kinetics/GABA/ligand_binding SUBSTRATE n 
addmsg /kinetics/GABA/ligand_binding /kinetics/GABA/Receptor REAC A B 
addmsg /kinetics/GABA/GABA /kinetics/GABA/ligand_binding SUBSTRATE n 
addmsg /kinetics/GABA/ligand_binding /kinetics/GABA/GABA REAC A B 
addmsg /kinetics/GABA/L_R /kinetics/GABA/ligand_binding PRODUCT n 
addmsg /kinetics/GABA/ligand_binding /kinetics/GABA/L_R REAC B A
addmsg /kinetics/GABA/Docked /kinetics/GABA/undocking SUBSTRATE n 
addmsg /kinetics/GABA/undocking /kinetics/GABA/Docked REAC A B 
addmsg /kinetics/GABA/RR_pool /kinetics/GABA/undocking PRODUCT n 
addmsg /kinetics/GABA/undocking /kinetics/GABA/RR_pool REAC B A
addmsg /kinetics/GABA/Docked /graphs/conc1/Docked.Co PLOT Co *Docked *54

enddump
 // End of dump
call /kinetics/GABA/vesicle_release/notes LOAD \
"High cooperativity, 4 or higher. Several refs: McDargh and O-Shaughnessy, BioRxiv 2021 Voleti, Jaczynska, Rizo, eLife 2020 Chen.... Scheller, Cell 1999"
complete_loading
