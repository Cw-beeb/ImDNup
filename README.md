# ImDNup


We propose ImDNup, an image-based DNA feature representation method to predict nucleosome positioning. The method encodes a DNA sequence into an image, called sequence2image vector, then applies a deep learning framework to automatically extract dominant patterns from sequence2image vectors for predicting nucleosome positioning of DNA sequences. 

The code is mainly written in Python (3.7) using  tensorflow2.2.3 .

The details about first three datasets can be found in the published paper:
Guo SH, Deng EZ, Xu LQ, Ding H, Lin H, Chen W, Chou KC: iNuc-PseKNC: a sequence-based predictor for predicting nucleosome positioning in genomes with pseudo k-tuple nucleotide composition. Bioinformatics 2014, 30(11):1522-1529.
The fourth dataset is introduced in the paper:
Chen W, Feng PM, Ding H, Lin H, Chou KC: Using deformation energy to analyze nucleosome positioning in genomes. Genomics 2016, 107(2-3):69-75.
