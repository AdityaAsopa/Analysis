# Whole Cell Patch Clamp Recording Setup: Theory and Operation

References:

1. Multiclamp 700B User manual
2. Axon Guide V Ed
3. Single Channel Recording, Sakmann and Neher, 1995  

-------------

## Recording synaptic currents in `Voltage Clamp`  

1. Use low resistance pipettes (2-4 MΩ)
2. Remove pipette offset
3. Neutralize Cp fast and Cp slow capacitance at gigaseal.
4. Compensate whole cell capacitance (Cm) after breaking in.
5. Measure Rs and discard if >25 MΩ
6. Compensate Rs with careful bandwidth, prediction, and correction settings.
7. Record baseline for 5 min to check for stable recording. Discard if shaky.
8. Keep checking and compensating for series resistance and noting its values.

Note: Low pipette resistance would also mean faster washout of cytoplasm. A trade-off worth balancing with right amount of Rs compensation with stable yet low resistance patches.

-------------

## Whole Cell Patch Clamp Steps  

### A. Electrode in bath

`VC`  
Pipette offset button: should cause the membrane current or holding current to become zero.
Seal test: A 10mV pulse is generated to determine the resistance. In the bath, it should come out to be equal to the pipette resistance (few MΩ)

`CC`  
Pipette offset: Pressing the pipette offset button should cause the pipette offset to become zero.
Tuning: Similar to seal test in VC. Injected a 1 nA pulse and measures the Vp to give pipette resistance value.

### B. Giga seal  

Transients appear and Rt ≈ 1 GΩ, therefore, steady state current is very low (< 10pA for a 10mV pulse)

Large amplitude transients appear at the start and end of the pulses. These are due to capacitances of patch electrode. These are bad as they can saturate the amplifier. There is a fast component and a slow component of this capacitance. These can be removed with Cp fast and Cp slow controls (either the Auto button or by manually changing C and tau values).
Cp fast: ≈ 5 pF, 1 us
Cp slow: ?
After compensation, the transients should disapper.

### C. Whole cell

`VC`
The steady state current, for a usual neuron will be around 100 pA for a 10mV pulse. That gives the input resistance of the whole cell (≈100 MΩ).

As soon as the seal is broken, the capacitance of the whole cell (Cm) comes into the circuit and new transients appear. These transients are much slower than the Cp fast and Cp slow transients that we compensated before breaking the seal. The capacitance of the cell is usually more than 30 pF and upto 200 pF depending on the branching.

These slow transients depend on the electrode resistance (now access resistance Ra) and cell capacitance (Cm).
We should cancel these slow transients also, for the following reasons:

1. They may saturate the amplifier.
2. This cancellation is required for the proper series resistance compensation.

This can be done in the "Whole Cell" section of VC tab. Auto and manual, both modes can be tried.

Note: If a fast transient reappears after whole cell capacitance compensation, that can be removed by Cp fast auto.

Note: Leak subtraction and Output zero sections are to remove the step currents due to voltage commands. Leak subtraction control works by subtracting a scaled version of the command voltage from the step current responses until the step current response becomes zero. This scaling would always be approximately equal to the input resistance of the cell. Output zero works to eliminate the offsets by applying a high pass filter. They are not recommended in whole cell recordings as there can be currents dependent on voltage of the cell.

Series Resistance Compensation:
Even after the elimination of the Cp fast, Cp slow, and Cm, one can observe that the current response to the voltage step rises slowly (≈1 ms). This is due to the series (aka access) resistance (Rs or Ra). The art of Rs compensation is to choose a combination of Bandwidth, Prediction and Correction that provides maximal compensation without oscillation. Described later.

`CC`  
In current clamp, Cm and Rs cause slow rise of membrane potential upon injection of a current step. The Rs is in series with Rm and therefore even before the Vm starts charging, an ohmic potential falls onto the Rs. This looks like a voltage step. To remove this in current clamp, bridge balance control is used. The amplifier subtracts a scaled version of command current from the measured Vm. This scaling factor is due to Rs and therefore will be approximately equal to it. Auto or manual both modes can be used.

Another factor is the capacitance of the electrode (Cp) that can cause errors. This can be removed by pipette capacitance neutralization. Over compensation can cause oscillations and therefore, pipette capacitance neutralization value should be carefully and gradually changed. (≈ < 3 pF)

### Pressure in the electrode

- Bath:   30 mbar,
- Before entering the tissue: 80 to 120 mbar,
- After entering the tissue:  30 to 50 mbar,
- Close to the cell:         -50 to -100 mbar,
- close to Gigaohm seal:      0 mbar,
- to break the seal:          80 mbar, 0.5s pulses

### Notes

- Fast capacitive spikes (cp fast) recorded are due to pipette and the enclosed membrane patch. They should be neurtralized before breaking into the cell.

- Gain settings should be selected for best possible resolution without hitting the saturation limit.  
    Feedback register (Rf) = 500 MΩ

    |         | Gain of 10  | Range     | Resolution |
    |---------|-------------|-----------|----------- |
    | VC      | 5 V/nA      |  ± 2 nA   | 0.03 pA/bit|
    | CC      | 100 mV/mV   | ± 100 mV  | 1.52 µV/bit|
  
- Breaking the patch:
  - Access to the cell can be seen as sudden increase in capacitive transients and,
  - depending on the input resistance of the cell, a shift in the current level or background noise.
  - Smaller access resistance => larger capacitive transients, but shorted duration
  - High levels of EGTA (10mM) buffers Ca+2 ions and prevents spontaneous increase in access resistance and resealing.  

- Capacitive trasient cancellation:
  - If the Cp fast was neutralized before breaking into the cell, then all the new capacitance will be due to the cell capacitance (Cm).
  - If cells have more complex geometry, there will be multiple capcitive time constants and neutralization will never be perfect or complete.

-------------

## Glossary

1. Bath resistance (Rb) is additive to the Rm and therefore is indistinguishable from Rm and acts as an access resistance component. It is usually very low (≈ 1 kΩ) but to keep it lower, larger surface area reference Ag/AgCl should be used.
For a passing of current of ≈ 1 nA, it would cause a potential drop of 1 µV.

2. Bridge Balance: Used to subtract voltage drop across pipette resistance (Rp) in current clamp. When a current is passed through the pipette into the cell, the cell Vm changes. This Vm is read at the end of the pipette as Vp. The passing current causes a voltage drop across the pipette and cause a voltage drop of:

   > Vdrop = I_inj * Rp  

    This unwanted voltage drop is added to the recorded potential. Bridge balance removes this drop so that only Vm is recorded.  

    When done in the bath: The bridge balance gives the value of resistance of the pipette.  

    ![Bridge Balance in Current Clamp](/notes_figures/bridge_balance_in_CClamp.png)

    When done is the cell: Pipette capacitance neutralization should be set the same time as the bridge balance. The pipette resistance and capacitance can change after the whole cell is established, so it is a good idea to monitor the bridge balance and pipette capacitance regularly while in the cell.  

    In compensating for electrode resistance, Bridge Balance is the current clamp analog of Rs Compensation in voltage clamp mode.  

    Electronically, it operates in a manner analogous to voltage clamp mode Leak Subtraction.

3. Capacitance compensation in Voltage clamp

    Electrode:  Cp fast component, represented as capacitance at headstage input and,
                Cp slow component, represented as a capacitor parallel to Rs
                This is done by supplying the required fast current through a capacitor (C1) instead of letting it pass through Rf.

    Cell     :  Whole cell capacitance (Cm)
                The time constant of whole cell capacitance transient is determined by the product of Cm and series resistance (Rs). If this becomes large, the transients can last longer causing distortion in cellular currents of interest. It is therefore, desirable to compensate for the cell capacitance. Also, whole cell capacitance compensation is required to compensate the series resistance.
                Our measure of cellular currents comes via feedback resistors. Whole cell capacitive compensation works by offloading the task of charging the cellular capacitor from Rf to a fast capacitor (C2).

    The capacitance and resistance values displayed by the whole cell compensation section in multiclamp commander give estimates of Cm and Rs respectively. These estimates are only good if Rs << Rm.

    ![Current signals before and after capacitance compensation](/notes_figures/capacitance_compensation_currents.png)

4. Capacitance Neutralization in Current Clamp

    In current clamp, the pipette capacitance is needed to be neutralized. This is easier if bridge balance is adjusted already to remove the pipette resistance. The option for pipette capacitance neutralization is in the current clamp tab. It should be gradually increased to avoid oscillations. (usual value ≈ 3pF)

5. External command sensitivity  
    For Rf = 500MΩ:  
    `VC`: ECS =  20 mV/V  
    `CC`: ECS = 400 pA/V

6. Feedback resistor (Rf)  
    `VC`: Determines the gain of headstage. Larger Rf means less noise, but at the risk of saturation if there are going to be larger currents in the cell. Rf of 500 MΩ is recommended for whole cell patching of neurons. I want to try 5GΩ. Since the amplifier can only create a maximum of 10 V step, this would mean that only upto 2 nA current can be passed if the Rf = 5GΩ.

    `CC`: Determines the maximal injectable current by the amplifier. Rf should be between 10xRin to 0.1xRin. For ex. for a hippocampal pyramidal cell (Rm = 150MΩ), the Rf can be between 15MΩ and 1500MΩ. Therefore, Rf = 500MΩ should be selected.

    `Note`: Changing Rf in CC will also change the external command sensitivity.

7. Grounding: Grounding bus should be connected to the "Signal Ground" at the back of the MCC module.

8. Headstage
    In `VC`
    Circuit is basically an I-V converter op-amp that converts current required to maintain Vp equal to Vcmd into voltage (Vo) that is measured. For single electrode voltage clamp

9. Liquid Junction Potential  
    Care must be taken to minimize junction-potential artifacts. When the pipette tip is immersed in the bath, a junction potential is generated. Its magnitude depends on the concentrations and mobilities of the ions in both the pipette solution and the bath.  

    The junction potential is usually nulled while the pipette tip is immersedin the bath. When the seal forms, the junction potential disappears orchanges its magnitude, and,consequently, the nulling voltage is no longercorrect. The junction potential also changes upon going whole cell. Theseeffects must be taken into account or offset errors will exist in eithersingle-channel or whole-cell current recordings.

    ![LJP Formula](/notes_figures/LJP_schematic.png)

    Calculated Values for Whole-cell measurements, with Ag/AgCl reference at 32°C

    `VC`  
    Ion         | z     | mobility  | Cpip | Cbath
    ------------|-------|-----------|------|-------
    Ca          |  2    | 0.4048    | 0    | 2
    Cl          | -1    | 1.0388    | 9    | 133.3
    Cs          |  1    | 1.05      | 130  | 0
    Gluconate   | -1    | 0.33      | 130  | 0
    H2PO        | -1    | 0.45      | 0    | 1.25
    HCO3        | -1    | 0.605     | 0    | 26
    HEPES       | -1    | 0.3       | 10   | 10
    K           |  1    | 1         | 0    | 2.7
    Mg          |  2    | 0.361     | 4    | 1.3
    Na          |  1    | 0.682     | 5.5  | 151.25
    H           |  1    | 4.76      | 130  | 0
    OH          | -1    | 2.7       | 130  | 0

    Junction Potential (original solution - pipette) = 17.4 mV
    Therefore:
    > Vm = Vp - (17.4) m V

    `CC`  
    Ion         | z     | mobility  | Cpip | Cbath
    ------------|-------|-----------|------|-------
    Ca          |  2    | 0.4048    | 0    | 2
    Cl          | -1    | 1.0388    | 9    | 133.3
    Gluconate   | -1    | 0.33      | 130  | 0
    H2PO        | -1    | 0.45      | 0    | 1.25
    HCO3        | -1    | 0.605     | 0    | 26
    HEPES       | -1    | 0.3       | 10   | 10
    K           |  1    | 1         | 130  | 2.7
    Mg          |  2    | 0.361     | 4    | 1.3
    Na          |  1    | 0.682     | 5.5  | 151.25

    Junction Potential (original solution - pipette) = 15.0 mV
    Therefore:
    > Vm = Vp - (15.0) mV

    Note: Ions like ATP, GTP, and EGTA are not accounted for due to their large size -> low mobility.

    Example of LJP Calculator in Clampex, based on Barry 1994:  

    ![LJP Calc](/notes_figures/LJP_calc_CC.png)

10. Voltage Clamp:  
    For single electrode VC, series resistance compensation is required. For the compensation to work, Rs should be lower than Rm. The voltage input to the op-amp is the voltage at the top of the pipette (Vp).

    > Vp = Vm + current induced voltage drop across the access resistance

    Access resistance (aka series resistance, Rs) = resistance due to pipette + cellular debris in the patch

    For large currents, the voltage drop on series resistance will be significant, thus causing insufficient voltage clamping, which will in turn cause wrong current injection to maintain the voltage clamp. Therefore, series resistance should be compensated. If Ra and Ra_eff are series resistance before and after compensation, then:

    > Vm  = Vcmd - Im * Ra_eff

    &

    > tau = [Ra_eff * Rm/ (Ra_eff + Rm)] * Cm

    if Rm >> Ra_eff:

    > tau ≈ Ra_eff * Cm

    Voltage Clamp Errors:
    - Space clamp: the injected current to maintain the cell at our desired voltage will radially spread into the cell and decay exponentially with a length constant (λ). Thus the Vm of the cell will be increasingly free of clamping away from the patch site. This means that currents produced by distant synapses in a branched cell are never seen by the amplifier and hence never recorded.
    - Point Clamp: The amplifier technically does not clamp the cell inside but the headstage input point due to the non-zero series resistance between that point and the cell. This makes series resistance compensation important and necessary. Point clamp error causes slower kinetics and offset in clamping. The latter is dependent on the current injected.

-------------

## Series resistance and Series resistance Compensation (SRC)  

Sources of series resistance:

- Pipette resistance
- Intracellular organelles and debris stuck in the patch
- Cellular debris that covers the cell membrane
- Bath solution and bath reference (minor)

Problems due to Rs:

- Steady state voltage errors: in steady state, a steady flow of current will cause a steady drop of voltage on Rs, with the direction of drop depending on direction of current.

    > Vm = Vcmd - Im * Rs

- Dynamic voltage errors: following a step change in Vcmd, the Vm will charge to Vcmd with an exponential time course according to:

    > tau = Rs * Cm

- Bandwidth errors: Rs and Cm together act like a low pass filter, with a -3dB cutoff given by:

    > f(-3dB) = 1/(2*pi*Rs*Cm)

    For a normal pyramidal cell, Cm ≈ 100 pF, if the Rs ≈ 20 MΩ, the -3dB frequency comes out to be **79.57 Hz**. This is too low to resolve most synaptic current waveforms properly. Therefore, it is always recommended to comepnsate the Rs.

## SRC = Correction + Prediction

Correction: Done by positive feedback. The Vcmd is increased by a signal proportional to the measured current. This increased command compensates for potential drop on the series resistance. The compensation beyond  ≈ 90% causes oscillations.

![Correction](/notes_figures/series_resistance_comp_correction.png)

Prediction: Done by supercharging, generally safer upto 95-98%  

![Prediction](/notes_figures/series_resistance_comp_prediction.png)

### Noise  

RMS  is better than P2P  

Sources:  
`VC`:  
    - High seal resistance -> low noise  
    - For larger cells, cellular noise dominates  
    - Low electrode capacitance --> low noise  
    - High electrode resistance  --> low noise, but it is better to have low resistance electrodes for having a better bandwidth  
    - High feedback resistor --> low noise  
    - Rs compensation increases noise  

`CC`:  
    - Low load resistance --> low noise  
    - Low capacitance --> low noise, but poor bandwidth  
    - Low Rf --> low noise, but Rf should be between 0.2 *Rm to 5* Rm (due to range of capacitance neutralization circuit)  
    - Increasing pipette cap neutralization --> more noise

-------------

## Other Notes

![Patch Clamp Stages](/notes_figures/patch_clamp_stages.png)

![Compensations](/notes_figures/Compensations_board_notes.png)
