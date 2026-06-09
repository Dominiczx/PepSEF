
['Polarity', 'Hydropathy_index']
phi_dict = {0: ['Ala', 'Phe', 'Ile', 'Met', 'Leu', 'Pro', 'Val'], \
    1: ['Gly', 'Trp'], \
    2: ['Cys'], \
    3: ['Asn', 'Gln', 'Ser', 'Thr', 'Tyr'], \
    4: ['Asp', 'Glu'], \
    5: ['Lys', 'His', 'Arg'], \
    6: ['other']     }

phi_att_dict = {}
for k, v in phi_dict.values():
    if k == 0:
        for pep in v:
            phi_att_dict[pep] = [0, 1]
    if k == 0:
        for pep in v:
            phi_att_dict[pep] = [0, 1]
    if k == 0:
        for pep in v:
            phi_att_dict[pep] = [0, 1]
