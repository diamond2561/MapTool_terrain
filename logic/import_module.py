import config
from PIL import Image
from PyQt6.QtWidgets import QFileDialog


def import_image(layout, text, image_display):
    Image.MAX_IMAGE_PIXELS = config.MAX_IMAGE_PIXELS

    path, _ = QFileDialog.getOpenFileName(
        layout,
        text,
        "",
        "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
    )
    if not path:
        return

    imported_image = Image.open(path)

    if imported_image.mode != "RGBA":
        imported_image = imported_image.convert("RGBA")

    # Если есть boundary — подгоняем размер
    if hasattr(layout, "boundary_image_display"):
        boundary_image = layout.boundary_image_display.get_image()
        if boundary_image and boundary_image.size != imported_image.size:
            imported_image = imported_image.resize(
                boundary_image.size,
                Image.NEAREST
            )

    image_display.set_image(imported_image)

    if hasattr(layout, "button_gen_prov"):
        layout.button_gen_prov.setEnabled(True)
