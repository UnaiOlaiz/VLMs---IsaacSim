The camera process calls the BentoML *ground_multi* call from the  [[BentoML Server]] endpoint. As mentioned, the detection results were not the best so I thought of implementing **Computer Vision algorithms** + some correcting offsets to enhance the predictions.

For each detectable object, I have designed a different script with different filters and procedure. However, I had to crop (get rid off certain distractive parts of the image) to help the VLM detect the target. 

These were the results: 
![[frankas_clean.png]]
![[Pasted image 20260416194133.png]]
![[result_clean.png]]

I decided to keep the basic white area to have just the basic detection, otherwise I would have to deal with texts and shadows (I had specific filters to get rid of the shadows). If I were to keep all the info, this would have been the output: ![[Pasted image 20260416194318.png]]

When it comes to making the VLM sure its targeting the right thing I established confidence counters and thresholds.

Plus, some essential functions to deal with the camera properties inside *IsaacSim*, such as the *focal pixel* and my screen resolution. The object detection results are stored in *JSON* formatted files. 

```python
data_save = {

"target_type": TARGET_TYPE,

"target_color": TARGET_COLOR,

"side": obj_id,

"world_pos": t_pos.tolist(),

"robot_world_pos": r_pos.tolist(),

"relative_pos": (t_pos - np.array(r_pos)).tolist(),

"camera_used": CAMERA_PATH,

"status": "calibrated_success"

}```
