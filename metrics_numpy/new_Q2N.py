import numpy as np

def q2n(I_GT, I_F, Q_blocks_size = 32, Q_shift = 32):

    N1,N2,N3 = I_GT.shape
    
    stepx = np.ceil(N1/Q_shift).astype(int)
    stepy = np.ceil(N2/Q_shift).astype(int)
     
    stepx = 1 if stepx <= 0 else stepx
    stepy = 1 if stepy <= 0 else stepy


    valori = np.zeros((stepx,stepy,N3))
    
    I_F[:,:,1:] = -I_F[:,:,1:]

    for j in range(stepx):
        for i in range(stepy):

            gt_block  = I_GT[(j*Q_shift):(j*Q_shift)+Q_blocks_size, (i*Q_shift):(i*Q_shift)+Q_blocks_size, : ]
            fus_block = I_F [(j*Q_shift):(j*Q_shift)+Q_blocks_size, (i*Q_shift):(i*Q_shift)+Q_blocks_size, : ]    
            valori[j,i,:] = onions_quality(gt_block, fus_block, Q_blocks_size)
          
    Q2n_index_map = np.sqrt(np.sum(valori**2, axis=2))
    Q2n_index = np.mean(Q2n_index_map)

    return Q2n_index, Q2n_index_map



def onions_quality(gt_block, fus_block, size1):
    
    N3 = gt_block.shape[2]
    size2 = size1
    
    """ Block normalization """
    gt_block, gt_block_means, gt_block_stds = norm_blocco(gt_block)

    fus_block[:,:,0] -= gt_block_means[0]  
    fus_block[:,:,1:] += gt_block_means[1:]   
    fus_block = np.where(gt_block_means==0, fus_block, fus_block/gt_block_stds)
    fus_block[:,:,0] += 1
    fus_block[:,:,1:] -= 1
    
    gt_block_means = np.mean(gt_block, axis=(0,1))
    fus_block_means = np.mean(fus_block, axis=(0,1))

    mod_q1 =  np.sqrt(np.sum(gt_block**2, axis=2))
    mod_q2 =  np.sqrt(np.sum(fus_block**2, axis=2))

    mod_q1m = np.sqrt(np.sum(gt_block_means**2))
    mod_q2m = np.sqrt(np.sum(fus_block_means**2))
    
    termine2 = mod_q1m * mod_q2m
    termine4 = mod_q1m**2 + mod_q2m**2

    mean_bias = 2*termine2/termine4

    int1 = (size1 * size2)/((size1 * size2)-1) * np.mean(mod_q1**2)
    int2 = (size1 * size2)/((size1 * size2)-1) * np.mean(mod_q2**2)
    termine3 = int1 + int2 - (size1 * size2)/((size1 * size2) - 1) * ((mod_q1m**2) + (mod_q2m**2))
    
    if (termine3==0):
        q = np.zeros((1,1,N3))
        q[:,:,-1] = mean_bias
    else:
        cbm = 2/termine3
        qu = onion_mult2D(gt_block, fus_block)
        qm = onion_mult(gt_block_means, fus_block_means)
        qv = (size1 * size2)/((size1 * size2)-1) * np.mean(qu, axis=(0,1))
        q = qv - ((size1 * size2)/((size1 * size2) - 1.0)) * qm
        q = q * mean_bias * cbm

    return q

def norm_blocco(blocks, epsilon= 2.2204 * 10**(-16)):
    
    block_means = np.mean(blocks, axis=(0,1))
    block_stds = np.std(blocks, ddof=1, axis=(0,1))   
    block_stds = np.where(block_stds==0, epsilon, block_stds)
    normalized_blocks = (blocks - block_means)/block_stds + 1

    return normalized_blocks, block_means, block_stds

def onion_mult2D(onion1,onion2):

    N3 = onion1.shape[2]

    if (N3 > 1):
        L = int(N3/2)
        a = onion1[:,:,0:L]
        b = onion1[:,:,L:]
        b[:,:,1:] = -b[:,:,1:]
        c = onion2[:,:,0:L]
        d = onion2[:,:,L:]
        d[:,:,1:] = -d[:,:,1:]

        if (N3 == 2):
            ris = np.concatenate((a*c-d*b,a*d+c*b),axis=2)
        else:
            ris1 = onion_mult2D(a, c)
            ris2 = onion_mult2D(d, np.concatenate((b[:,:,:1],-b[:,:,1:]),axis=2))
            ris3 = onion_mult2D(np.concatenate((a[:,:,:1],-a[:,:,1:]),axis=2), d)
            ris4 = onion_mult2D(c, b)
            aux1 = ris1 - ris2
            aux2 = ris3 + ris4
            ris = np.concatenate((aux1,aux2), axis=2)
    else:
        ris = onion1 * onion2   

    return ris

def onion_mult(onion1,onion2):
    N = onion1.size
    
    if (N > 1):      
        L = int(N/2)          
        a = onion1[0:L]
        b = onion1[L:]
        b[1:] = -b[1:]
        c = onion2[0:L]
        d = onion2[L:]
        d[1:] = -d[1:]

        if (N == 2):
            ris = np.concatenate([a*c-d*b, a*d+c*b])
        else:
            ris1 = onion_mult(a, c)
            ris2 = onion_mult(d, np.concatenate((b[:1],-b[1:])))
            ris3 = onion_mult(np.concatenate((a[:1],-a[1:])), d)
            ris4 = onion_mult(c, b)  
            aux1 = ris1 - ris2
            aux2 = ris3 + ris4  
            ris = np.concatenate((aux1, aux2))
    else:
        ris = onion1 * onion2
    return ris
