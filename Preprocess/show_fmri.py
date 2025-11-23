import imageio
import argparse
import numpy as np
import nibabel as nib

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--dura", type=float, default=0.034)
parser.add_argument("--dim", type=int, default=4)

args = parser.parse_args()

def scale_data(data):
    p10 = np.percentile(data, 10)
    p99 = np.percentile(data, 99.5)
    data[data<p10] = p10
    data -= p10
    data[data>p99] = p99
    data /= p99
    data *= 255

    return data

def main(args):
    data = nib.load(args.data)
    data = nib.as_closest_canonical(data)
    data = data.get_fdata()
    # data-=data.min()
    # data/=data.max()
    # data*=255
    data = scale_data(data)

    data=data.astype(np.uint8)
    frames = []
    
    for idx in range(data.shape[3]):
        if args.dim==1:
            frames.append(data[data.shape[0]//2,:,::-1,idx].T)
        elif args.dim==2:
            frames.append(data[:,data.shape[1]//2,::-1,idx].T)
        elif args.dim==3:
            frames.append(data[:,::-1,data.shape[2]//2,idx].T)
        elif args.dim==4:
            fig=np.zeros([data.shape[1]+data.shape[2], data.shape[0]+data.shape[1]], dtype=np.uint8)

            fig[:data.shape[2],:data.shape[0]] = data[:,data.shape[1]//2,::-1,idx].T
            fig[data.shape[2]:data.shape[2]+data.shape[1],:data.shape[0]] = data[:,::-1,data.shape[2]//2,idx].T
            fig[:data.shape[2],data.shape[0]:data.shape[0]+data.shape[1]] = data[data.shape[0]//2,:,::-1,idx].T

            frames.append(fig)
        else:
            print('dim must in [1, 2, 3, 4]')
            return

    #iio.imwrite(args.out+'.gif', np.stack(frames, axis=0), mode="I")
    imageio.mimsave(args.out+'.gif', frames, 'GIF', duration=args.dura)

main(args)

