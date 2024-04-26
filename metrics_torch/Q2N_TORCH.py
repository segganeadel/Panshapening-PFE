import torch


def q2n_torch(I_GT:torch.Tensor, I_F:torch.Tensor, Q_blocks_size=32, Q_shift=32):
    """
    Parameters
    ----------
    I_GT : torch.Tensor
        Ground truth images of shape (N, C, H, W)
    I_F : torch.Tensor
        Fusion images of shape (N, C, H, W)
    Q_blocks_size : int, optional
        Size of the blocks, by default 32
    Q_shift : int, optional
        Shift of the blocks, by default 32

    Returns
    -------
    tuple[Tensor, Tensor]
        Q2n indexes and Q2n maps of shape (N,) and (N, H, W)
    """

    # TODO: change the code to use the same input format without the need to permute
    I_GT = I_GT.permute(0,2,3,1)
    I_F = I_F.permute(0,2,3,1)
    
    indexes = []
    maps = []

    # TODO: change the code to lose the need to iterate over the batch
    for i in range(I_GT.shape[0]):
        index, q2n_map = q2n(I_GT[i], I_F[i], Q_blocks_size, Q_shift)
        indexes.append(index)
        maps.append(q2n_map)

    indexes = torch.tensor(indexes)
    maps = torch.stack(maps)

    return indexes, maps


def q2n(I_GT, I_F, Q_blocks_size = 32, Q_shift = 32):

    N1,N2,N3 = I_GT.shape
    
    # TODO: is this really the best way to do this ?
    steps = torch.ceil(torch.tensor(I_GT.shape[:2])/Q_shift).int()

    stepx = 1 if steps[0] <= 0 else steps[0]
    stepy = 1 if steps[1] <= 0 else steps[1]


    #Calculate wether it needs pdding
    est1 = (stepx - 1) * Q_shift + Q_blocks_size - N1
    est2 = (stepy - 1) * Q_shift + Q_blocks_size - N2

    #Mirror padding if needed
    
    # TODO: vectorize padding on H x W (probably using torch.nn.functional.pad or change the way it works)
    # padding lower anf right most sides, could it be done around the whole image ?
    if (est1 != 0) or (est2 != 0):
        refref = []
        fusfus = []
        
        for i in range(N3):
         
            a1 = torch.squeeze(I_GT[:,:,0]) 
            ia1 = torch.zeros((N1+est1,N2+est2))
            ia1[0:N1,0:N2] = a1

            ia1[:,N2:N2+est2] = ia1[:,N2-1:N2-est2-1:-1]
            ia1[N1:N1+est1,:] = ia1[N1-1:N1-est1-1:-1,:]

            if i == 0:
                refref = ia1
            elif i == 1:
                refref = torch.concatenate((refref[:,:,None],ia1[:,:,None]),axis=2)
            else:
                refref = torch.concatenate((refref,ia1[:,:,None]),axis=2)
            
            if (i < (N3-1)):
                I_GT = I_GT[:,:,1:I_GT.shape[2]]
            
        I_GT = refref
            
        for i in range(N3):
            a2 = torch.squeeze(I_F[:,:,0])
            
            ia2 = torch.zeros((N1+est1,N2+est2))
            ia2[0:N1,0:N2] = a2
            ia2[:,N2:N2+est2] = ia2[:,N2-1:N2-est2-1:-1]
            ia2[N1:N1+est1,:] = ia2[N1-1:N1-est1-1:-1,:]
            
            if i == 0:
                fusfus = ia2
            elif i == 1:
                fusfus = torch.concatenate((fusfus[:,:,None],ia2[:,:,None]),axis=2)
            else:
                fusfus = torch.concatenate((fusfus,ia2[:,:,None]),axis=2)
            
            if (i < (N3-1)):
                I_F = I_F[:,:,1:I_F.shape[2]]
                           
        I_F = fusfus
      
    # If number of channels are not power of 2 then pad zeros to make it power of 2 with padding by zeros
    # TODO: vectorize padding channels
    log_N3 = torch.log2(torch.tensor(N3))
    if (((torch.ceil(log_N3)) - log_N3) != 0):
        Ndif = (2**(torch.ceil(log_N3))) - torch.tensor(N3)
        Ndif = int(Ndif)
        dif = torch.zeros((N1,N2,Ndif))
        #dif = torch.uint16(dif)
        I_GT = torch.concatenate((I_GT, dif), axis = 2)
        I_F = torch.concatenate((I_F, dif), axis = 2)
 

    # Vectorisation attempt 
    # TODO: use torch.nn.functional.unfold

    # x_indexes_begin = torch.arange(stepx)[:, None] * Q_shift
    # x_indexes_end = (torch.arange(stepx)[:, None] + 1) * Q_shift
    # y_indexes_begin = torch.arange(stepy)[None, :] * Q_shift
    # y_indexes_end = (torch.arange(stepy)[None, :] + 1) * Q_shift

    # gt_blocks = I_GT[ x_indexes_begin : x_indexes_end, y_indexes_begin : y_indexes_end, : ]
    # fus_blocks = I_F[ x_indexes_begin : x_indexes_end, y_indexes_begin : y_indexes_end, : ]

    # print(gt_blocks.shape)
    # print(fus_blocks.shape)
    # #valori = onions_quality(gt_blocks, fus_blocks, Q_blocks_size)

    valori = torch.zeros((stepx,stepy,N3))

    for j in range(stepx):
        for i in range(stepy):
            gt_block  = I_GT[(j*Q_shift):(j*Q_shift)+Q_blocks_size, (i*Q_shift):(i*Q_shift)+Q_blocks_size, : ]
            fus_block = I_F [(j*Q_shift):(j*Q_shift)+Q_blocks_size, (i*Q_shift):(i*Q_shift)+Q_blocks_size, : ]        
            valori[j,i,:] = onions_quality(gt_block, fus_block, Q_blocks_size)
            
    Q2n_index_map = torch.sqrt(torch.sum(valori**2, axis=2))
    Q2n_index = torch.mean(Q2n_index_map)

    return Q2n_index, Q2n_index_map


def onions_quality(gt_block, fus_block, block_size):

    # does simply flipping the [1:] channels without using concatenate affect calculations ?
    fus_block = torch.concatenate((fus_block[:,:,:1], -fus_block[:,:,1:]), axis=2)
    N3 = gt_block.shape[2]
    
    """ Block normalization """
    gt_block, gt_block_means_not_norm, gt_block_stds_not_norm = nomalize_block(gt_block)

    fus_block[:,:,0] -= gt_block_means_not_norm[0]  
    fus_block[:,:,1:] += gt_block_means_not_norm[1:]   
    fus_block = torch.where(gt_block_means_not_norm==0, fus_block, fus_block/gt_block_stds_not_norm)
    fus_block[:,:,0] += 1
    fus_block[:,:,1:] -= 1
    
    gt_block_means_not_norm = torch.mean(gt_block, axis=(0,1))
    fus_block_means = torch.mean(fus_block, axis=(0,1))

    mod_q1 =  torch.sqrt(torch.sum(gt_block**2, axis=2))
    mod_q2 =  torch.sqrt(torch.sum(fus_block**2, axis=2))

    mod_q1m = torch.sqrt(torch.sum(gt_block_means_not_norm**2))
    mod_q2m = torch.sqrt(torch.sum(fus_block_means**2))
    
    termine2 = mod_q1m * mod_q2m
    termine4 = mod_q1m**2 + mod_q2m**2

    mean_bias = 2*termine2/termine4

    block_area = block_size**2
    int1 = block_area/(block_area-1) * torch.mean(mod_q1**2)
    int2 = block_area/(block_area-1) * torch.mean(mod_q2**2)
    termine3 = int1 + int2 - block_area/(block_area - 1) * ((mod_q1m**2) + (mod_q2m**2))
    
    if (termine3==0):
        q = torch.zeros((1,1,N3))
        q[:,:,-1] = mean_bias
    else:
        cbm = 2/termine3
        qu = onion_mult2D(gt_block, fus_block)
        qm = onion_mult(gt_block_means_not_norm, fus_block_means)
        qv = block_area/(block_area-1) * torch.mean(qu, axis=(0,1))
        q = qv - (block_area/(block_area - 1.0)) * qm
        q = q * mean_bias * cbm

    return q

def nomalize_block(blocks, epsilon= 2.2204 * 10**(-16)):
    
    block_means = torch.mean(blocks, axis=(0,1))
    block_stds = torch.std(blocks, correction=1, axis=(0,1))   
    block_stds = torch.where(block_stds==0, epsilon, block_stds)
    normalized_blocks = (blocks - block_means)/block_stds + 1

    return normalized_blocks, block_means, block_stds

def onion_mult2D(onion1,onion2):
    # TODO: need a way to remove recursion
    N3 = onion1.shape[2]

    if (N3 > 1):
        L = int(N3/2)
        a = onion1[:,:,0:L]
        b = onion1[:,:,L:]
        b = torch.concatenate((b[:,:,:1], -b[:,:,1:]), axis=2)
        c = onion2[:,:,0:L]
        d = onion2[:,:,L:]
        d = torch.concatenate((d[:,:,:1],-d[:,:,1:]), axis=2)

        if (N3 == 2):
            ris = torch.concatenate((a*c-d*b, a*d+c*b),axis=2)
        else:
            ris1 = onion_mult2D(a, c)
            ris2 = onion_mult2D(d, torch.concatenate((b[:,:,:1], -b[:,:,1:]), axis=2))
            ris3 = onion_mult2D(torch.concatenate((a[:,:,:1], -a[:,:,1:]), axis=2), d)
            ris4 = onion_mult2D(c, b)
            aux1 = ris1 - ris2
            aux2 = ris3 + ris4
            ris = torch.concatenate((aux1, aux2), axis=2)
    else:
        ris = onion1 * onion2   

    return ris

def onion_mult(onion1,onion2):
    # TODO: need a way to remove recursion
    N = onion1.shape[0]
    
    if (N > 1):      
        L = int(N/2)          
        a = onion1[0:L]
        b = onion1[L:]
        b = torch.concatenate((b[:1], -b[1:]))
        c = onion2[0:L]
        d = onion2[L:]
        d = torch.concatenate((d[:1], -d[1:]))

        if (N == 2):
            ris = torch.concatenate([a*c-d*b, a*d+c*b])
        else:
            ris1 = onion_mult(a, c)
            ris2 = onion_mult(d, torch.concatenate((b[:1], -b[1:])))
            ris3 = onion_mult(torch.concatenate((a[:1], -a[1:])), d)
            ris4 = onion_mult(c, b)  
            aux1 = ris1 - ris2
            aux2 = ris3 + ris4  
            ris = torch.concatenate((aux1, aux2))
    else:
        ris = onion1 * onion2
    return ris
