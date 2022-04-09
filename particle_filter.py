import json
import os
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# change IDs to your IDs.
ID1 = "203135058"
ID2 = "203764170"

ID = "HW3_{0}_{1}".format(ID1, ID2)
RESULTS = 'results'
os.makedirs(RESULTS, exist_ok=True)
IMAGE_DIR_PATH = "Images"

# SET NUMBER OF PARTICLES
N = 100

# Initial Settings
s_initial = [297,    # x center
             139,    # y center
              16,    # half width
              43,    # half height
               0,    # velocity x
               0]    # velocity y

# Dynamic Model
A = np.identity(6)
A[0, 4] = 1  # add Vx to Xc
A[1, 5] = 1  # add Vy to Yc


def predict_particles(s_prior: np.ndarray) -> np.ndarray:
    """Progress the prior state with time and add noise.

    Note that we explicitly did not tell you how to add the noise.
    We allow additional manipulations to the state if you think these are necessary.

    Args:
        s_prior: np.ndarray. The prior state.
    Return:
        state_drifted: np.ndarray. The prior state after drift (applying the motion model) and adding the noise.
    """
    s_prior = s_prior.astype(float)
    state_drifted = A@s_prior
    # TODO: think which noise to add
    state_drifted[0] = state_drifted[0] + np.random.uniform(0, 5, (1, N))
    state_drifted[1] = state_drifted[1] + np.random.uniform(0, 5, (1, N))
    state_drifted[4] = state_drifted[4] + np.random.normal(0, 5, (1, N))
    state_drifted[5] = state_drifted[5] + np.random.normal(0, 2, (1, N))
    state_drifted = state_drifted.astype(int)
    return state_drifted


def compute_normalized_histogram(image: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Compute the normalized histogram using the state parameters.

    Args:
        image: np.ndarray. The image we want to crop the rectangle from.
        state: np.ndarray. State candidate.

    Return:
        hist: np.ndarray. histogram of quantized colors.
    """
    state = np.floor(state)
    state = state.astype(int)

    work_image = image.copy()
    # crop the image to the required rectangle
    x_c = state[0]
    y_c = state[1]
    half_w = state[2]
    half_h = state[3]
    start_col = max(0,x_c - half_w)
    end_col = min(x_c + half_w, image.shape[1])
    start_row = max(0,y_c - half_h)
    end_row = min(y_c + half_h, image.shape[0])
    image_sub_portion = work_image[start_row:end_row, start_col:end_col, :]

    # quantization to 4-bits
    for start_grey_level in range(0, 255, 32):
        image_sub_portion[(image_sub_portion >= start_grey_level) & (image_sub_portion < start_grey_level+32)] = int(start_grey_level/32)

    # calc hist
    hist = np.zeros((16, 16, 16))
    b = 0
    g = 1
    r = 2
    image_b = image_sub_portion[:, :, b]
    image_g = image_sub_portion[:, :, g]
    image_r = image_sub_portion[:, :, r]

    # TODO: maybe optimize it
    for b_grey_level in range(16):
        for g_grey_level in range(16):
            for r_grey_level in range(16):
                hist[b_grey_level][g_grey_level][r_grey_level] = ((image_b == b_grey_level) & (image_g == g_grey_level) & (image_r == r_grey_level)).sum()

    hist = np.reshape(hist, 16 * 16 * 16)
    # normalize
    if sum(hist) != 0:
        hist = hist/sum(hist)
    else:
        # TODO: debug
        print(f'row: [{start_row}:{end_row}] col: [{start_col},{end_col}]')
    return hist


def sample_particles(previous_state: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """Sample particles from the previous state according to the cdf.

    If additional processing to the returned state is needed - feel free to do it.

    Args:
        previous_state: np.ndarray. previous state, shape: (6, N)
        cdf: np.ndarray. cummulative distribution function: (N, )

    Return:
        s_next: np.ndarray. Sampled particles. shape: (6, N)
    """
    S_next = np.zeros(previous_state.shape)
    for particl in range(N):
        r = np.random.uniform(0, 1)
        j = np.argmax(cdf >= r)
        S_next[:, particl] = previous_state[:, j]
    return S_next


def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate Bhattacharyya Distance between two histograms p and q.

    Args:
        p: np.ndarray. first histogram.
        q: np.ndarray. second histogram.

    Return:
        distance: float. The Bhattacharyya Distance.
    """

    # TODO: make sure the implementation is OK
    exp_sum = 0
    for i in range(4096):
        exp_sum += np.sqrt(p[i] * q[i])
    distance = np.exp(20 * exp_sum)

    return distance


def show_particles(image: np.ndarray, state: np.ndarray, W: np.ndarray, frame_index: int, ID: str,
                  frame_index_to_mean_state: dict, frame_index_to_max_state: dict,
                  ) -> tuple:
    fig, ax = plt.subplots(1)
    image = image[:,:,::-1]
    plt.imshow(image)
    plt.title(ID + " - Frame mumber = " + str(frame_index))

    # TODO: debug
    for i in range(N):
        (x_par, y_par, w_par, h_par) = (state[0,i]-state[2,i], state[1,i]-state[3,i], 2*state[2,i], 2*state[3,i])
        rect = patches.Rectangle((x_par, y_par), w_par, h_par, linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect)

    # Avg particle box
    state0_avg = np.average(state[0], weights=W)
    state1_avg = np.average(state[1], weights=W)
    state2_avg = np.average(state[2], weights=W)
    state3_avg = np.average(state[3], weights=W)
    (x_avg, y_avg, w_avg, h_avg) = (state0_avg-state2_avg, state1_avg-state3_avg, 2*state2_avg, 2*state3_avg)

    rect = patches.Rectangle((x_avg, y_avg), w_avg, h_avg, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    # calculate Max particle box
    (x_max, y_max, w_max, h_max) = (0, 0, 0, 0)
    """ DELETE THE LINE ABOVE AND:
        INSERT YOUR CODE HERE."""

    rect = patches.Rectangle((x_max, y_max), w_max, h_max, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    #plt.show(block=False)

    fig.savefig(os.path.join(RESULTS, ID + "-" + str(frame_index) + ".png"))
    frame_index_to_mean_state[frame_index] = [float(x) for x in [x_avg, y_avg, w_avg, h_avg]]
    frame_index_to_max_state[frame_index] = [float(x) for x in [x_max, y_max, w_max, h_max]]
    return frame_index_to_mean_state, frame_index_to_max_state


def main():
    state_at_first_frame = np.matlib.repmat(s_initial, N, 1).T
    # TODO: make sure the mu,sigma re fine
    '''    noise = np.random.randint(0, 100, (6, N))
        for i in [0,1,4,5]:
            state_at_first_frame[i] = state_at_first_frame[i] + noise[i]'''
    S = predict_particles(state_at_first_frame)

    # LOAD FIRST IMAGE
    image = cv2.imread(os.path.join(IMAGE_DIR_PATH, "001.png"))

    # COMPUTE NORMALIZED HISTOGRAM
    q = compute_normalized_histogram(image, s_initial)

    # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
    # YOU NEED TO FILL THIS PART WITH CODE:
    W = []
    # TODO: make sure the q stays the same and only p is calculated each iteration
    # TODO: make sre this is correct after we will have the previous functions
    # TODO: maybe optimize it takes a lot of time.
    for i in range(N):
        p = compute_normalized_histogram(image, S[:, i])
        W.append(bhattacharyya_distance(p, q))
    W = np.array(W)
    W = W / W.sum()

    C = [W[0]]
    for i in range(N - 1):
        C.append(C[i] + W[i + 1])
    C = np.array(C)

    images_processed = 1

    # MAIN TRACKING LOOP
    image_name_list = os.listdir(IMAGE_DIR_PATH)
    image_name_list.sort()
    frame_index_to_avg_state = {}
    frame_index_to_max_state = {}
    for image_name in image_name_list[1:]:

        S_prev = S

        # LOAD NEW IMAGE FRAME
        image_path = os.path.join(IMAGE_DIR_PATH, image_name)
        current_image = cv2.imread(image_path)

        # SAMPLE THE CURRENT PARTICLE FILTERS
        S_next_tag = sample_particles(S_prev, C)

        # PREDICT THE NEXT PARTICLE FILTERS (YOU MAY ADD NOISE
        S = predict_particles(S_next_tag)

        # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
        # YOU NEED TO FILL THIS PART WITH CODE:
        # TODO: check this copy+paste
        W = []
        for i in range(N):
            p = compute_normalized_histogram(current_image, S[:, i])
            W.append(bhattacharyya_distance(p, q))
        W = np.array(W)
        W = W / W.sum()

        C = [W[0]]
        for i in range(N - 1):
            C.append(C[i] + W[i + 1])
        C = np.array(C)

        # CREATE DETECTOR PLOTS
        images_processed += 1
        # TODO: change this to the original code
        if 0 == images_processed%1:
            frame_index_to_avg_state, frame_index_to_max_state = show_particles(
                current_image, S, W, images_processed, ID, frame_index_to_avg_state, frame_index_to_max_state)

    with open(os.path.join(RESULTS, 'frame_index_to_avg_state.json'), 'w') as f:
        json.dump(frame_index_to_avg_state, f, indent=4)
    with open(os.path.join(RESULTS, 'frame_index_to_max_state.json'), 'w') as f:
        json.dump(frame_index_to_max_state, f, indent=4)


if __name__ == "__main__":
    main()
