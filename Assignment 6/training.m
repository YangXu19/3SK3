clear;
clc;
[imgs labels] = readMNIST("train-images-idx3-ubyte", "train-labels-idx1-ubyte", 10, 0002); 
list = [0:9]
for k = 1 : 10
      for RandomMatrix = randi([-10,9],[400,1])
          w{k} = transpose(RandomMatrix);
          x{k} = reshape(imgs(:,:,1),[],1);
          z{k} = dot(w{k},x{k})
          if z{k} >= 0  
            y{k} = 1
          else
            y{k} = -1
            if labels(k) == list(k)
                t = 1
            else
                t = -1
            end  
          end
      end
    end