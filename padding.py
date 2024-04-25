
    # Calculate wether it needs pdding

    # est1 = (stepx - 1) * Q_shift + Q_blocks_size - N1
    # est2 = (stepy - 1) * Q_shift + Q_blocks_size - N2

    # #Mirror padding if needed
    
    # if (est1 != 0) or (est2 != 0):
    #   refref = []
    #   fusfus = []
      
    #   for i in range(N3):
          
    #       a1 = np.squeeze(I_GT[:,:,0]) 
    #       ia1 = np.zeros((N1+est1,N2+est2))
    #       ia1[0:N1,0:N2] = a1

    #       ia1[:,N2:N2+est2] = ia1[:,N2-1:N2-est2-1:-1]
    #       ia1[N1:N1+est1,:] = ia1[N1-1:N1-est1-1:-1,:]

    #       if i == 0:
    #           refref = ia1
    #       elif i == 1:
    #           refref = np.concatenate((refref[:,:,None],ia1[:,:,None]),axis=2)
    #       else:
    #           refref = np.concatenate((refref,ia1[:,:,None]),axis=2)
          
    #       if (i < (N3-1)):
    #           I_GT = I_GT[:,:,1:I_GT.shape[2]]
          
    #   I_GT = refref
            
    #   for i in range(N3):
    #       a2 = np.squeeze(I_F[:,:,0])
          
    #       ia2 = np.zeros((N1+est1,N2+est2))
    #       ia2[0:N1,0:N2] = a2
    #       ia2[:,N2:N2+est2] = ia2[:,N2-1:N2-est2-1:-1]
    #       ia2[N1:N1+est1,:] = ia2[N1-1:N1-est1-1:-1,:]
          
    #       if i == 0:
    #           fusfus = ia2
    #       elif i == 1:
    #           fusfus = np.concatenate((fusfus[:,:,None],ia2[:,:,None]),axis=2)
    #       else:
    #           fusfus = np.concatenate((fusfus,ia2[:,:,None]),axis=2)
          
    #       if (i < (N3-1)):
    #           I_F = I_F[:,:,1:I_F.shape[2]]
          
    #   I_F = fusfus
      
    # If number of channels are not power of 2 then pad zeros to make it power of 2

    # if (((math.ceil(math.log2(N3))) - math.log2(N3)) != 0):
    #     Ndif = (2**(math.ceil(math.log2(N3)))) - N3
    #     dif = np.zeros((N1,N2,Ndif))
    #     #dif = np.uint16(dif)
    #     I_GT = np.concatenate((I_GT, dif), axis = 2)
    #     I_F = np.concatenate((I_F, dif), axis = 2)
 