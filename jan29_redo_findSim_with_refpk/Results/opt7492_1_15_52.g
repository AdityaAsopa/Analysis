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
DEFAULT_VOL = 1.729191019041669e-21
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
simundump geometry /kinetics/geometry 0 1.729191019041669e-21 3 sphere  "" white black 9 5 0
simundump group /kinetics/glu 0 blue green x 0 0 "" defaultfile \
  defaultfile.g 0 0 0 6 5 0
simundump kpool /kinetics/glu/Ca 0 0.0 0 0 0 0.08330947876582084 0 0 1.0413432997198127 0 /kinetics/geometry 62 yellow 7 3 0
simundump kpool /kinetics/glu/Ca_ext 0 0.0 0 0 0 0.08330947876582084 0 0 1.0413432997198127 4 /kinetics/geometry 52 yellow 11 3 0
simundump kpool /kinetics/glu/RR_pool 0 0.0 0 0 0 0.26033078795936326 0 0 1.0413432997198127 0 /kinetics/geometry 1 yellow 5 1 0
simundump kpool /kinetics/glu/vesicle_pool 0 0.0 0 0 0 0.26787880576525 0 0 1.0413432997198127 4 /kinetics/geometry 27 yellow 11 -1 0
simundump kpool /kinetics/glu/glu 0 0.0 0 0 0 0.0 0 0 1.0413432997198127 0 /kinetics/geometry 7 yellow 11 1 0
simundump kpool /kinetics/glu/Docked 0 0.0 0 0 0 0.0 0 0 1.0413432997198127 0 /kinetics/geometry 54 blue 9 1 0
simundump kpool /kinetics/glu/Ca_RR 0 0.0 0 0 0 0.0 0 0 1.0413432997198127 0 /kinetics/geometry 51 blue 7 1 0
simundump kpool /kinetics/glu/Receptor 0 0.0 0 0 0 15.166318445662931 0 0 1.0413432997198127 0 /kinetics/geometry 12 yellow 13 3 0
simundump kpool /kinetics/glu/L_R 0 0.0 0 0 0 0.0 0 0 1.0413432997198127 0 /kinetics/geometry 23 yellow 13 1 0
simundump kreac /kinetics/glu/remove_Ca 0 14.404491490977065 14.404491490977065 "" white yellow 9 4 0
simundump kreac /kinetics/glu/remove 0 2212.0584534522122 0.0 "" white yellow 9 0 0
simundump kreac /kinetics/glu/replenish_vesicle 0 14.373149992529548 14.373149992529548 "" white yellow 9 -2 0
simundump kreac /kinetics/glu/vesicle_release 0 1.4962374215559573 0.0 "" white yellow 10 2 0
simundump kreac /kinetics/glu/Ca_bind_RR 0 2.0076860838777737 10.829487575971113 "" white blue 6 2 0
simundump kreac /kinetics/glu/docking 0 296.7017772899774 0.0 "" white blue 8 2 0
simundump kreac /kinetics/glu/ligand_binding 0 731.0236932429019 230.14861101349305 "" white blue 12 2 0
simundump kreac /kinetics/glu/undocking 0 5.0 0.0 "" white blue 7 0 0
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
addmsg /kinetics/glu/Ca /kinetics/glu/remove_Ca SUBSTRATE n 
addmsg /kinetics/glu/remove_Ca /kinetics/glu/Ca REAC A B 
addmsg /kinetics/glu/Ca /kinetics/glu/remove_Ca SUBSTRATE n 
addmsg /kinetics/glu/remove_Ca /kinetics/glu/Ca REAC A B 
addmsg /kinetics/glu/Ca_ext /kinetics/glu/remove_Ca PRODUCT n 
addmsg /kinetics/glu/remove_Ca /kinetics/glu/Ca_ext REAC B A
addmsg /kinetics/glu/Ca_ext /kinetics/glu/remove_Ca PRODUCT n 
addmsg /kinetics/glu/remove_Ca /kinetics/glu/Ca_ext REAC B A
addmsg /kinetics/glu/glu /kinetics/glu/remove SUBSTRATE n 
addmsg /kinetics/glu/remove /kinetics/glu/glu REAC A B 
addmsg /kinetics/glu/Ca /kinetics/glu/remove PRODUCT n 
addmsg /kinetics/glu/remove /kinetics/glu/Ca REAC B A
addmsg /kinetics/glu/Ca /kinetics/glu/remove PRODUCT n 
addmsg /kinetics/glu/remove /kinetics/glu/Ca REAC B A
addmsg /kinetics/glu/vesicle_pool /kinetics/glu/remove PRODUCT n 
addmsg /kinetics/glu/remove /kinetics/glu/vesicle_pool REAC B A
addmsg /kinetics/glu/vesicle_pool /kinetics/glu/replenish_vesicle SUBSTRATE n 
addmsg /kinetics/glu/replenish_vesicle /kinetics/glu/vesicle_pool REAC A B 
addmsg /kinetics/glu/RR_pool /kinetics/glu/replenish_vesicle PRODUCT n 
addmsg /kinetics/glu/replenish_vesicle /kinetics/glu/RR_pool REAC B A
addmsg /kinetics/glu/Ca /kinetics/glu/vesicle_release SUBSTRATE n 
addmsg /kinetics/glu/vesicle_release /kinetics/glu/Ca REAC A B 
addmsg /kinetics/glu/Ca /kinetics/glu/vesicle_release SUBSTRATE n 
addmsg /kinetics/glu/vesicle_release /kinetics/glu/Ca REAC A B 
addmsg /kinetics/glu/Docked /kinetics/glu/vesicle_release SUBSTRATE n 
addmsg /kinetics/glu/vesicle_release /kinetics/glu/Docked REAC A B 
addmsg /kinetics/glu/glu /kinetics/glu/vesicle_release PRODUCT n 
addmsg /kinetics/glu/vesicle_release /kinetics/glu/glu REAC B A
addmsg /kinetics/glu/Ca /kinetics/glu/Ca_bind_RR SUBSTRATE n 
addmsg /kinetics/glu/Ca_bind_RR /kinetics/glu/Ca REAC A B 
addmsg /kinetics/glu/Ca /kinetics/glu/Ca_bind_RR SUBSTRATE n 
addmsg /kinetics/glu/Ca_bind_RR /kinetics/glu/Ca REAC A B 
addmsg /kinetics/glu/RR_pool /kinetics/glu/Ca_bind_RR SUBSTRATE n 
addmsg /kinetics/glu/Ca_bind_RR /kinetics/glu/RR_pool REAC A B 
addmsg /kinetics/glu/Ca_RR /kinetics/glu/Ca_bind_RR PRODUCT n 
addmsg /kinetics/glu/Ca_bind_RR /kinetics/glu/Ca_RR REAC B A
addmsg /kinetics/glu/Ca_RR /kinetics/glu/docking SUBSTRATE n 
addmsg /kinetics/glu/docking /kinetics/glu/Ca_RR REAC A B 
addmsg /kinetics/glu/Ca /kinetics/glu/docking PRODUCT n 
addmsg /kinetics/glu/docking /kinetics/glu/Ca REAC B A
addmsg /kinetics/glu/Ca /kinetics/glu/docking PRODUCT n 
addmsg /kinetics/glu/docking /kinetics/glu/Ca REAC B A
addmsg /kinetics/glu/Docked /kinetics/glu/docking PRODUCT n 
addmsg /kinetics/glu/docking /kinetics/glu/Docked REAC B A
addmsg /kinetics/glu/glu /kinetics/glu/ligand_binding SUBSTRATE n 
addmsg /kinetics/glu/ligand_binding /kinetics/glu/glu REAC A B 
addmsg /kinetics/glu/Receptor /kinetics/glu/ligand_binding SUBSTRATE n 
addmsg /kinetics/glu/ligand_binding /kinetics/glu/Receptor REAC A B 
addmsg /kinetics/glu/L_R /kinetics/glu/ligand_binding PRODUCT n 
addmsg /kinetics/glu/ligand_binding /kinetics/glu/L_R REAC B A
addmsg /kinetics/glu/Docked /kinetics/glu/undocking SUBSTRATE n 
addmsg /kinetics/glu/undocking /kinetics/glu/Docked REAC A B 
addmsg /kinetics/glu/RR_pool /kinetics/glu/undocking PRODUCT n 
addmsg /kinetics/glu/undocking /kinetics/glu/RR_pool REAC B A
addmsg /kinetics/glu/Docked /graphs/conc1/Docked.Co PLOT Co *Docked *54

enddump
 // End of dump
call /kinetics/glu/vesicle_release/notes LOAD \
"High cooperativity, 4 or higher. Several refs: McDargh and O-Shaughnessy, BioRxiv 2021 Voleti, Jaczynska, Rizo, eLife 2020 Chen.... Scheller, Cell 1999"
complete_loading
