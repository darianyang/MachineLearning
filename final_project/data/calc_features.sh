#!/bin/bash

PDB=1a43
OUT=${PDB}_features.dat
# for nc calcs
if [ $PDB == "1a43" ] ; then
    REF=xtal
elif [ $PDB == "2kod" ] ; then
    REF=nmr
fi

# cpptraj input
CMD="     parm ${PDB}_dry.prmtop \n" 
CMD="$CMD trajin ${PDB}_1us_1000i.nc 1 1000 1 \n"
CMD="$CMD reference 2kod_solv.pdb name [nmr] \n"
CMD="$CMD reference 1a43_solv.pdb :* name [xtal] \n"
CMD="$CMD parm hex_leap.pdb [hexparm] \n"
CMD="$CMD reference hex_leap.pdb parm [hexparm] name [hex] \n"
CMD="$CMD parm pent_leap.pdb [pentparm] \n"
CMD="$CMD reference pent_leap.pdb parm [pentparm] name [pent] \n"
CMD="$CMD autoimage \n"

# NMR ref rms calcs
CMD="$CMD rms Fit_M2_NMR :94-163&!@H= ref [nmr] \n"
CMD="$CMD rms RMS_M1_NMR :6-75&!@H= ref [nmr] nofit mass time 1 out $OUT \n"
CMD="$CMD rms RMS_H9M1_NMR :36-49&!@H= nofit ref [nmr] out $OUT mass time 1 \n"

CMD="$CMD rms Fit_M1_NMR :6-75&!@H= ref [nmr] \n"
CMD="$CMD rms RMS_M2_NMR :94-163&!@H= ref [nmr] nofit mass time 1 out $OUT \n"
CMD="$CMD rms RMS_H9M2_NMR :124-137&!@H= nofit ref [nmr] out $OUT mass time 1 \n"

CMD="$CMD rms RMS_Heavy_NMR :6-75,94-163&!@H= ref [nmr] mass time 1 out $OUT \n"
CMD="$CMD rms RMS_Backbone_NMR :6-75,94-163@CA,C,O,N ref [nmr] mass time 1 out $OUT \n"
CMD="$CMD rms RMS_Dimer_Int_NMR :7,8,34,35,37,38,41,42,45,46,49,95,96,122,123,125,126,129,130,133,134,137&!@H= ref [nmr] mass time 1 out $OUT \n"
CMD="$CMD rms RMS_Key_Int_NMR :38,41,42,126,129,130&!@H= ref [nmr] mass time 1 out $OUT \n"

# XTAL ref rms calcs
CMD="$CMD rms Fit_M2_XTAL :94-163&!@H= ref [xtal] \n"
CMD="$CMD rms RMS_M1_XTAL :6-75&!@H= ref [xtal] nofit mass time 1 out $OUT \n"
CMD="$CMD rms RMS_H9M1_XTAL :36-49&!@H= nofit ref [xtal] out $OUT mass time 1 \n"

CMD="$CMD rms Fit_M1_XTAL :6-75&!@H= ref [xtal] \n"
CMD="$CMD rms RMS_M2_XTAL :94-163&!@H= ref [xtal] nofit mass time 1 out $OUT \n"
CMD="$CMD rms RMS_H9M2_XTAL :124-137&!@H= nofit ref [xtal] out $OUT mass time 1 \n"

CMD="$CMD rms RMS_Heavy_XTAL :6-75,94-163&!@H= ref [xtal] mass time 1 out $OUT \n"
CMD="$CMD rms RMS_Backbone_XTAL :6-75,94-163@CA,C,O,N ref [xtal] mass time 1 out $OUT \n"
CMD="$CMD rms RMS_Dimer_Int_XTAL :7,8,34,35,37,38,41,42,45,46,49,95,96,122,123,125,126,129,130,133,134,137&!@H= ref [xtal] mass time 1 out $OUT \n"
CMD="$CMD rms RMS_Key_Int_XTAL :38,41,42,126,129,130&!@H= ref [xtal] mass time 1 out $OUT \n"

# HEX ref rms calcs
CMD="$CMD rms Fit_M2_HEX :94-163&!@H= ref [hex] \n"
CMD="$CMD rms RMS_M1_HEX :6-75&!@H= ref [hex] nofit mass time 1 out $OUT \n"
CMD="$CMD rms RMS_H9M1_HEX :36-49&!@H= nofit ref [hex] out $OUT mass time 1 \n"

CMD="$CMD rms Fit_M1_HEX :6-75&!@H= ref [hex] \n"
CMD="$CMD rms RMS_M2_HEX :94-163&!@H= ref [hex] nofit mass time 1 out $OUT \n"
CMD="$CMD rms RMS_H9M2_HEX :124-137&!@H= nofit ref [hex] out $OUT mass time 1 \n"

CMD="$CMD rms RMS_Heavy_HEX :6-75,94-163&!@H= ref [hex] mass time 1 out $OUT \n"
CMD="$CMD rms RMS_Backbone_HEX :6-75,94-163@CA,C,O,N ref [hex] mass time 1 out $OUT \n"
CMD="$CMD rms RMS_Dimer_Int_HEX :7,8,34,35,37,38,41,42,45,46,49,95,96,122,123,125,126,129,130,133,134,137&!@H= ref [hex] mass time 1 out $OUT \n"
CMD="$CMD rms RMS_Key_Int_HEX :38,41,42,126,129,130&!@H= ref [hex] mass time 1 out $OUT \n"

# PENT ref rms calcs
CMD="$CMD rms Fit_M2_PENT :94-163&!@H= ref [pent] \n"
CMD="$CMD rms RMS_M1_PENT :6-75&!@H= ref [pent] nofit mass time 1 out $OUT \n"
CMD="$CMD rms RMS_H9M1_PENT :36-49&!@H= nofit ref [pent] out $OUT mass time 1 \n"

CMD="$CMD rms Fit_M1_PENT :6-75&!@H= ref [pent] \n"
CMD="$CMD rms RMS_M2_PENT :94-163&!@H= ref [pent] nofit mass time 1 out $OUT \n"
CMD="$CMD rms RMS_H9M2_PENT :124-137&!@H= nofit ref [pent] out $OUT mass time 1 \n"

CMD="$CMD rms RMS_Heavy_PENT :6-75,94-163&!@H= ref [pent] mass time 1 out $OUT \n"
CMD="$CMD rms RMS_Backbone_PENT :6-75,94-163@CA,C,O,N ref [pent] mass time 1 out $OUT \n"
CMD="$CMD rms RMS_Dimer_Int_PENT :7,8,34,35,37,38,41,42,45,46,49,95,96,122,123,125,126,129,130,133,134,137&!@H= ref [pent] mass time 1 out $OUT \n"
CMD="$CMD rms RMS_Key_Int_PENT :38,41,42,126,129,130&!@H= ref [pent] mass time 1 out $OUT \n"

# dimer angle calc, vector based
CMD="$CMD vector D1 :1-75@CA,C,O,N :39@CA,C,O,N  \n"
CMD="$CMD vector D2 :89-163@CA,C,O,N :127@CA,C,O,N  \n"
CMD="$CMD vectormath vec1 D1 vec2 D2 out $OUT name c2_angle dotangle \n"
# using backbone N of: T186 M1 --- M1 and M2 V181 --- T186 M2
CMD="$CMD angle helix_angle_3pt :43@N :38,126@N :131@N out $OUT mass \n"
# calc dimer orientation angle using vectors
CMD="$CMD vector O1 :18-22@CA,C,O,N :46-49@CA,C,O,N \n"
CMD="$CMD vector O2 :18-22@CA,C,O,N :134-137@CA,C,O,N \n"
CMD="$CMD vector O3 :106-110@CA,C,O,N :46-49@CA,C,O,N \n"
CMD="$CMD vector O4 :106-110@CA,C,O,N :134-137@CA,C,O,N \n"
CMD="$CMD vectormath vec1 O1 vec2 O2 out $OUT name o_angle_m1 dotangle \n"
CMD="$CMD vectormath vec1 O3 vec2 O4 out $OUT name o_angle_m2 dotangle \n"

# rog, sasa, dssp, contacts 
CMD="$CMD radgyr RoG :1-176 out $OUT mass nomax \n"
CMD="$CMD radgyr RoG-cut :6-75,94-163 out $OUT mass nomax \n"
CMD="$CMD surf Total_SASA :1-176 out $OUT \n"
CMD="$CMD nativecontacts name Num_Inter_Contacts :1-88&!@H= :89-176&!@H= byresidue distance 4.5 out $OUT ref [$REF] \n"
CMD="$CMD nativecontacts name Num_Intra_Contacts :1-176&!@H= byresidue distance 4.5 out $OUT ref [$REF] \n"

# calculate multiple distances
# calc distance between (COM) both O eps of E175 and H eps 1 of W184
CMD="$CMD distance M1-E175-Oe_M2-W184-He1 :32@OE1,OE2 :129@HE1 out $OUT \n"
CMD="$CMD distance M2-E175-Oe_M1-W184-He1 :120@OE1,OE2 :41@HE1 out $OUT \n"
# calc distance between (COM) both O eps of E175 and HG1 of T148
CMD="$CMD distance M1-E175-Oe_M1-T148-HG1 :32@OE1,OE2 :5@HG1 out $OUT \n"
CMD="$CMD distance M2-E175-Oe_M2-T148-HG1 :120@OE1,OE2 :93@HG1 out $OUT \n"
# calc inter monomer distance using COMs of M1 and M2
CMD="$CMD distance M1-M2-COM :6-75&!@H= :94-163&!@H= out $OUT \n"
# calc inter monomer distance using bottom of the helix
CMD="$CMD distance M1-M2-L46 :46@N :134@N out $OUT \n"

# dihedral angles of E175
CMD="$CMD multidihedral phi resrange 32-32 out $OUT \n"
CMD="$CMD multidihedral psi resrange 32-32 out $OUT \n"
CMD="$CMD multidihedral dihtype chi1:N:CA:CB:CG "
CMD="$CMD               dihtype chi2:CA:CB:CG:CD "
CMD="$CMD               dihtype chi3:CB:CG:CD:OE1 "
CMD="$CMD               resrange 32-32"
CMD="$CMD               out $OUT \n"
CMD="$CMD multidihedral phi resrange 120-120 out $OUT \n"
CMD="$CMD multidihedral psi resrange 120-120 out $OUT \n"
CMD="$CMD multidihedral dihtype chi1:N:CA:CB:CG "
CMD="$CMD               dihtype chi2:CA:CB:CG:CD "
CMD="$CMD               dihtype chi3:CB:CG:CD:OE1 "
CMD="$CMD               resrange 120-120"
CMD="$CMD               out $OUT \n"

# done
CMD="$CMD go \n"

echo -e "$CMD" | cpptraj
