import numpy as np

def q2n(I_GT, I_F, Q_blocks_size = 32, Q_shift = 32):

    N1,N2,N3 = I_GT.shape
    
    steps = np.ceil(np.array(I_GT.shape[:2])/Q_shift).astype(int)

    stepx = 1 if steps[0] <= 0 else steps[0]
    stepy = 1 if steps[1] <= 0 else steps[1]






    #Calculate wether it needs pdding

    est1 = (stepx - 1) * Q_shift + Q_blocks_size - N1
    est2 = (stepy - 1) * Q_shift + Q_blocks_size - N2

    #Mirror padding if needed
    
    if (est1 != 0) or (est2 != 0):
        refref = []
        fusfus = []
        
        for i in range(N3):
            
            a1 = np.squeeze(I_GT[:,:,0]) 
            ia1 = np.zeros((N1+est1,N2+est2))
            ia1[0:N1,0:N2] = a1

            ia1[:,N2:N2+est2] = ia1[:,N2-1:N2-est2-1:-1]
            ia1[N1:N1+est1,:] = ia1[N1-1:N1-est1-1:-1,:]

            if i == 0:
                refref = ia1
            elif i == 1:
                refref = np.concatenate((refref[:,:,None],ia1[:,:,None]),axis=2)
            else:
                refref = np.concatenate((refref,ia1[:,:,None]),axis=2)
            
            if (i < (N3-1)):
                I_GT = I_GT[:,:,1:I_GT.shape[2]]
            
        I_GT = refref
            
        for i in range(N3):
            a2 = np.squeeze(I_F[:,:,0])
            
            ia2 = np.zeros((N1+est1,N2+est2))
            ia2[0:N1,0:N2] = a2
            ia2[:,N2:N2+est2] = ia2[:,N2-1:N2-est2-1:-1]
            ia2[N1:N1+est1,:] = ia2[N1-1:N1-est1-1:-1,:]
            
            if i == 0:
                fusfus = ia2
            elif i == 1:
                fusfus = np.concatenate((fusfus[:,:,None],ia2[:,:,None]),axis=2)
            else:
                fusfus = np.concatenate((fusfus,ia2[:,:,None]),axis=2)
            
            if (i < (N3-1)):
                I_F = I_F[:,:,1:I_F.shape[2]]
                           
        I_F = fusfus
      
    # If number of channels are not power of 2 then pad zeros to make it power of 2

    # if (((math.ceil(math.log2(N3))) - math.log2(N3)) != 0):
    #     Ndif = (2**(math.ceil(math.log2(N3)))) - N3
    #     dif = np.zeros((N1,N2,Ndif))
    #     #dif = np.uint16(dif)
    #     I_GT = np.concatenate((I_GT, dif), axis = 2)
    #     I_F = np.concatenate((I_F, dif), axis = 2)
 



    # x_indexes_begin = np.arange(stepx)[:, None] * Q_shift
    # x_indexes_end = (np.arange(stepx)[:, None] + 1) * Q_shift
    # y_indexes_begin = np.arange(stepy)[None, :] * Q_shift
    # y_indexes_end = (np.arange(stepy)[None, :] + 1) * Q_shift

    # gt_blocks = I_GT[ x_indexes_begin : x_indexes_end, y_indexes_begin : y_indexes_end, : ]
    # fus_blocks = I_F[ x_indexes_begin : x_indexes_end, y_indexes_begin : y_indexes_end, : ]

    # print(gt_blocks.shape)
    # print(fus_blocks.shape)
    # #valori = onions_quality(gt_blocks, fus_blocks, Q_blocks_size)

    valori = np.zeros((stepx,stepy,N3))

    for j in range(stepx):
        for i in range(stepy):
            gt_block  = I_GT[(j*Q_shift):(j*Q_shift)+Q_blocks_size, (i*Q_shift):(i*Q_shift)+Q_blocks_size, : ]
            fus_block = I_F [(j*Q_shift):(j*Q_shift)+Q_blocks_size, (i*Q_shift):(i*Q_shift)+Q_blocks_size, : ]        
            valori[j,i,:] = onions_quality(gt_block, fus_block, Q_blocks_size)
            
    Q2n_index_map = np.sqrt(np.sum(valori**2, axis=2))
    Q2n_index = np.mean(Q2n_index_map)

    return Q2n_index, Q2n_index_map


def onions_quality(gt_block, fus_block, block_size):

    fus_block = np.concatenate((fus_block[:,:,:1], -fus_block[:,:,1:]), axis=2)
    N3 = gt_block.shape[2]
    
    """ Block normalization """
    gt_block, gt_block_means_not_norm, gt_block_stds_not_norm = norm_blocco(gt_block)

    fus_block[:,:,0] -= gt_block_means_not_norm[0]  
    fus_block[:,:,1:] += gt_block_means_not_norm[1:]   
    fus_block = np.where(gt_block_means_not_norm==0, fus_block, fus_block/gt_block_stds_not_norm)
    fus_block[:,:,0] += 1
    fus_block[:,:,1:] -= 1
    
    gt_block_means_not_norm = np.mean(gt_block, axis=(0,1))
    fus_block_means = np.mean(fus_block, axis=(0,1))

    mod_q1 =  np.sqrt(np.sum(gt_block**2, axis=2))
    mod_q2 =  np.sqrt(np.sum(fus_block**2, axis=2))

    mod_q1m = np.sqrt(np.sum(gt_block_means_not_norm**2))
    mod_q2m = np.sqrt(np.sum(fus_block_means**2))
    
    termine2 = mod_q1m * mod_q2m
    termine4 = mod_q1m**2 + mod_q2m**2

    mean_bias = 2*termine2/termine4
    block_area = block_size**2
    int1 = block_area/(block_area-1) * np.mean(mod_q1**2)
    int2 = block_area/(block_area-1) * np.mean(mod_q2**2)
    termine3 = int1 + int2 - block_area/(block_area - 1) * ((mod_q1m**2) + (mod_q2m**2))
    
    if (termine3==0):
        q = np.zeros((1,1,N3))
        q[:,:,-1] = mean_bias
    else:
        cbm = 2/termine3
        qu = onion_mult2D(gt_block, fus_block)
        qm = onion_mult(gt_block_means_not_norm, fus_block_means)
        qv = block_area/(block_area-1) * np.mean(qu, axis=(0,1))
        q = qv - (block_area/(block_area - 1.0)) * qm
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
        b = np.concatenate((b[:,:,:1], -b[:,:,1:]), axis=2)
        c = onion2[:,:,0:L]
        d = onion2[:,:,L:]
        d = np.concatenate((d[:,:,:1],-d[:,:,1:]), axis=2)

        if (N3 == 2):
            ris = np.concatenate((a*c-d*b, a*d+c*b),axis=2)
        else:
            ris1 = onion_mult2D(a, c)
            ris2 = onion_mult2D(d, np.concatenate((b[:,:,:1], -b[:,:,1:]), axis=2))
            ris3 = onion_mult2D(np.concatenate((a[:,:,:1], -a[:,:,1:]), axis=2), d)
            ris4 = onion_mult2D(c, b)
            aux1 = ris1 - ris2
            aux2 = ris3 + ris4
            ris = np.concatenate((aux1, aux2), axis=2)
    else:
        ris = onion1 * onion2   

    return ris

def onion_mult(onion1,onion2):
    N = onion1.size
    
    if (N > 1):      
        L = int(N/2)          
        a = onion1[0:L]
        b = onion1[L:]
        b = np.concatenate((b[:1], -b[1:]))
        c = onion2[0:L]
        d = onion2[L:]
        d = np.concatenate((d[:1], -d[1:]))

        if (N == 2):
            ris = np.concatenate([a*c-d*b, a*d+c*b])
        else:
            ris1 = onion_mult(a, c)
            ris2 = onion_mult(d, np.concatenate((b[:1], -b[1:])))
            ris3 = onion_mult(np.concatenate((a[:1], -a[1:])), d)
            ris4 = onion_mult(c, b)  
            aux1 = ris1 - ris2
            aux2 = ris3 + ris4  
            ris = np.concatenate((aux1, aux2))
    else:
        ris = onion1 * onion2
    return ris
