# Resize the images to 50x50
find . -name "*.png" -exec mogrify -resize 50x50 {} \;