import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from networks.net_factory import net_factory

# Set the paths directly in the code
model_name = 'unet'
num_classes = 4
image_path = './image_4.png'  # Set your image path here
model_path = './model/BCP/ACDC_BCP_7_labeled/self_train/unet_best_model.pth'  # Set your model path here
output_path = './image_4_mask_bcp10%.png'  # Set your output mask path here

def segment_single_image(image_path, model, output_path, num_classes=4):
    image = Image.open(image_path).convert('L')
    original_size = image.size
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(image).unsqueeze(0).cuda()

    model.eval()
    with torch.no_grad():
        out_main = model(input_tensor)
        if len(out_main) > 1:
            out_main = out_main[0]
        out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()

    # Debugging: Print unique values in the output mask
    unique_values = np.unique(out)
    print(f"Unique values in the output mask: {unique_values}")

    # Map the values to higher intensity
    out_mapped = np.zeros_like(out)
    out_mapped[out == 1] = 85
    out_mapped[out == 2] = 170
    out_mapped[out == 3] = 255

    # Resize the output mask back to the original image size
    mask = Image.fromarray(out_mapped.astype(np.uint8))
    mask = mask.resize(original_size, Image.NEAREST)

    # Save the mask as a PNG image
    mask.save(output_path)
    print(f"Segmentation mask saved to {output_path}")

if __name__ == '__main__':
    # Load the model
    net = net_factory(net_type=model_name, in_chns=1, class_num=num_classes)
    net.load_state_dict(torch.load(model_path))
    net.cuda()

    # Segment the image
    segment_single_image(image_path, net, output_path, num_classes=num_classes)
