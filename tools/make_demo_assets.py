"""One-off helper: populate demo identities and query assets for smoke tests.

Demo-only: uses OpenCV sample imagery (Lena/Messi) as clearly-labeld demo
identities so the pipeline can be validated before real consented reference
photos are added. Replace database/ contents with real authorized photos
before any demonstration (PROJECT_SPEC section 18).
"""

import cv2

lena = cv2.imread("input/images/lena_test.jpg")
messi = cv2.imread("input/images/messi_test.jpg")

cv2.imwrite("database/person_001/mirrored.jpg", cv2.flip(lena, 1))
cv2.imwrite("database/person_002/front.jpg", messi)
cv2.imwrite("input/images/lena_flip_query.jpg", cv2.flip(lena, 1))
cv2.imwrite("input/images/scenery_test.jpg", cv2.rotate(lena, cv2.ROTATE_90_CLOCKWISE) // 6 * 6)
print("demo assets written")
