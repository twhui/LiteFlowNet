function AEE = endPointErr(predict, label, val_map)

if size(predict)~=size(label)
    error('shape not equal');
end

if size(predict, 3)~=2 || size(label, 3)~=2
    error('need two-channel flow field');
end

EPE = sqrt((predict(:,:,1) - label(:,:,1)).^2 + (predict(:,:,2) - label(:,:,2)).^2); 
EPE(val_map == 0) = 0;

AEE = sum(sum(EPE))/sum(sum(val_map == 1));