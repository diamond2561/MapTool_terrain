from PyQt6.QtWidgets import QFileDialog
import csv


def export_image(parent_layout, image, text):
    if image is None:
        return

    try:
        path, _ = QFileDialog.getSaveFileName(
            parent_layout, text, "", "PNG Files (*.png)"
        )
        if path:
            image.save(path)
    except Exception as error:
        print(f"Error saving image: {error}")


def export_provinces_csv(main_layout):
    metadata = getattr(main_layout, "province_data", None)
    if not metadata:
        print("No province data to export.")
        return

    path, _ = QFileDialog.getSaveFileName(
        main_layout, "Export Province CSV", "", "CSV Files (*.csv)"
    )
    if not path:
        return

    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=';')
            w.writerow([
                "province_id", "R", "G", "B",
                "province_type", "terrain", "x", "y"
            ])

            for d in metadata:
                w.writerow([
                    d["province_id"],
                    d["R"], d["G"], d["B"],
                    d["province_type"],
                    d["terrain"],
                    f'="{d["x"]:.2f}"',
                    f'="{d["y"]:.2f}"'
                ])
    except Exception as e:
        print("Error saving province data:", e)
