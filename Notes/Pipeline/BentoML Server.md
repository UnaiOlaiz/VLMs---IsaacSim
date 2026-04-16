Bento uses **Qwen2-VL-2B-Instruct** model.
(I will write the current workflow, the old strategies in the official documentation)
The VLM must detect the cubes, pallets, and the franka arms (2).

- **VLMServiceIsaac**: service class, has multiple functions such as *hallucination filter*, the *infer* function.
- **ground_multi**: api call function that will receive the target color parameter, the prompt string, the target type (robot/cube/pallet) and a custom helping prompt.

Useful strategies:
- I was using a single detection function, the current function must filter from the detected objetcs.
- The VLM would fail most of the times (not detect or fail with a large error), so I implemented several custom prompts for each objects, and the median of the predicted coordinates for each prompt will be the final prediction. **Much better results**. 

**The VLM receives a pair of Image-Prompt**.