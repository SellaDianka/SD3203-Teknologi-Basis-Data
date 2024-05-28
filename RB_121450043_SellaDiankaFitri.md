Nama : Sella Dianka Fitri
NIM : 121450043
Kelas : RB

# KODE 1
python
import numpy as np
import pickle
from pathlib import Path
data_dir = Path("D:/Downloads/Lenovo/cifar-10-python/cifar-10-batches-py")
def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict
images, labels = [], []
for batch in data_dir.glob("data_batch_*"):
    batch_data = unpickle(batch)
    for i, flat_im in enumerate(batch_data[b"data"]):
        im_channels = []
        # Each image is flattened, with channels in order of R, G, B
        for j in range(3):
            im_channels.append(
                flat_im[j * 1024 : (j + 1) * 1024].reshape((32, 32))
            )
        # Reconstruct the original image
        images.append(np.dstack((im_channels)))
        # Save the label
        labels.append(batch_data[b"labels"][i])
print("Loaded CIFAR-10 training set:")
print(f" - np.shape(images)     {np.shape(images)}")
print(f" - np.shape(labels)     {np.shape(labels)}")
```
{.output .stream .stdout}
    Loaded CIFAR-10 training set:
     - np.shape(images)     (50000, 32, 32, 3)
     - np.shape(labels)     (50000,)

# ANALISIS
Kode diatas digunakan untuk memuat dataset CIFAR-10 dari file batch yang disimpan dalam direktori data_dir. Pertama, kita , kemudian path menuju dataset ditentukan. Fungsi unpickle didefinisikan untuk membaca file pickle CIFAR-10, dan file batch diproses satu per satu untuk mengekstrak gambar dan label. Setiap gambar akan diubah dari format array 1D yang diperluas menjadi gambar RGB, dan hasil akhirnya adalah array gambar dengan bentuk (50000, 32, 32, 3) dan array label dengan bentuk (50000,), yang diverifikasi dengan mencetak bentuk array tersebut.

Selanjutnya kita akan melakukan setup untuk menyimpan gambar kedalam disk

# Kode
pip install Pillow
conda install -c conda-forge pillow

# Analisis
Untuk menyimpan gambar didalam disk maka kita perlu melakukan install library yang dibutuhkan

# Kode
pip install lmdb
conda install -c conda-forge python-lmdb

# Analisis
LMDB langsung memetakan data ke dalam memori, memungkinkan pengembalian penunjuk langsung ke alamat memori untuk kunci dan nilai tanpa perlu menyalin data ke memori seperti yang dilakukan oleh sebagian besar database lainnya. Untuk menyimpan dan mengambil gambar menggunakan format LMDB, beberapa langkah berikut perlu dilakukan dalam shell Python pada perangkat yang digunakan.

# Kode
pip install h5py
conda install -c conda-forge h5py

# Analisis
Perintah `pip install h5py` digunakan untuk menginstal library `h5py` melalui pip, yang merupakan package manager untuk Python. Alternatifnya, `conda install -c conda-forge h5py` menginstal `h5py` menggunakan conda dari channel conda-forge, yang sering digunakan untuk mendapatkan versi terbaru dan teruji dari paket.

# Kode menyimpan single image
```python
from pathlib import Path
disk_dir = Path("data/disk/")
lmdb_dir = Path("data/lmdb/")
hdf5_dir = Path("data/hdf5/")
disk_dir.mkdir(parents=True, exist_ok=True)
lmdb_dir.mkdir(parents=True, exist_ok=True)
hdf5_dir.mkdir(parents=True, exist_ok=True)

# Analisis
Kode tersebut membuat tiga direktori untuk menyimpan data gambar dalam format disk, LMDB, dan HDF5. Direktori ini dibuat dengan menggunakan `mkdir` yang memastikan bahwa semua direktori induk dibuat jika belum ada dan menghindari error jika direktori sudah ada. Langkah ini memastikan struktur direktori siap sebelum menyimpan gambar.

# Menyimpan ke Disk

```python
from PIL import Image
import csv
def store_single_disk(image, image_id, label):
    """ Stores a single image as a .png file on disk.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    Image.fromarray(image).save(disk_dir / f"{image_id}.png")
    with open(disk_dir / f"{image_id}.csv", "wt") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        writer.writerow([label])

# Analisis
Fungsi `store_single_disk` menyimpan satu gambar sebagai file .png di disk dan menuliskan labelnya ke file .csv yang sesuai. Gambar diubah menjadi format yang dapat disimpan menggunakan PIL, sedangkan label disimpan dalam file CSV dengan penulisan minimal menggunakan modul csv. Proses ini memastikan bahwa setiap gambar dan labelnya dapat diakses secara independen dalam format yang mudah dibaca.

# Menyimpan ke LMBD
```python
class CIFAR_Image:
    def __init__(self, image, label):
        # Dimensions of image for reconstruction - not really necessary
        # for this dataset, but some datasets may include images of
        # varying sizes
        self.channels = image.shape[2]
        self.size = image.shape[:2]
        self.image = image.tobytes()
        self.label = label
    def get_image(self):
        """ Returns the image as a numpy array. """
        image = np.frombuffer(self.image, dtype=np.uint8)
        return image.reshape(*self.size, self.channels)

# Analisis
Kelas `CIFAR_Image` dirancang untuk menyimpan gambar CIFAR-10 bersama labelnya dalam format byte yang mudah disimpan dalam database. Metode `__init__` mengonversi gambar menjadi byte dan menyimpan dimensi serta label gambar, sementara metode `get_image` mengonversi kembali byte menjadi array numpy sesuai dimensi asli gambar. Ini memungkinkan penyimpanan dan pemulihan gambar secara efisien dalam database yang tidak mendukung tipe data kompleks secara langsung.

# Tutup Lingkungan LMBD
```python
import lmdb
import pickle
def store_single_lmdb(image, image_id, label):
    """ Stores a single image to a LMDB.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    map_size = image.nbytes * 10
    # Create a new LMDB environment
    env = lmdb.open(str(lmdb_dir / f"single_lmdb"), map_size=map_size)
    # Start a new write transaction
    with env.begin(write=True) as txn:
        # All key-value pairs need to be strings
        value = CIFAR_Image(image, label)
        key = f"{image_id:08}"
        txn.put(key.encode("ascii"), pickle.dumps(value))
    env.close()
```

# Analisis
Fungsi `store_single_lmdb` menyimpan satu gambar CIFAR-10 ke dalam database LMDB. Dengan membuat lingkungan LMDB baru dan memulai transaksi penulisan, fungsi ini mengonversi gambar dan label menjadi format byte menggunakan kelas `CIFAR_Image`, lalu menyimpan pasangan kunci-nilai yang sesuai. Setelah transaksi selesai, lingkungan LMDB ditutup untuk memastikan semua data tersimpan dengan benar.

# Menyimpan ke HDF5
```python
import h5py
def store_single_hdf5(image, image_id, label):
    """ Stores a single image to an HDF5 file.
        Parameters:
        ---------------
        image       image array, (32, 32, 3) to be stored
        image_id    integer unique ID for image
        label       image label
    """
    # Create a new HDF5 file
    file = h5py.File(hdf5_dir / f"{image_id}.h5", "w")
    # Create a dataset in the file
    dataset = file.create_dataset(
        "image", np.shape(image), h5py.h5t.STD_U8BE, data=image
    )
    meta_set = file.create_dataset(
        "meta", np.shape(label), h5py.h5t.STD_U8BE, data=label
    )
    file.close()
``python
_store_single_funcs = dict(
    disk=store_single_disk, lmdb=store_single_lmdb, hdf5=store_single_hdf5
)
```

# Analisis
Fungsi `store_single_hdf5` menyimpan satu gambar CIFAR-10 ke dalam file HDF5, menciptakan dataset untuk gambar dan label, serta menutup file setelah penyimpanan selesai. Variabel `_store_single_funcs` digunakan untuk menyimpan semua fungsi penyimpanan gambar dalam format disk, LMDB, dan HDF5 sebagai kamus untuk penggunaan selanjutnya.

# Eksperimen Storing Single Images
```python
from timeit import timeit
store_single_timings = dict()
for method in ("disk", "lmdb", "hdf5"):
    t = timeit(
        "_store_single_funcs[method](image, 0, label)",
        setup="image=images[0]; label=labels[0]",
        number=1,
        globals=globals(),
    )
    store_single_timings[method] = t
    print(f"Method: {method}, Time usage: {t}")
```

{Output}
Method: disk, Time usage: 0.4261889010325463
Method: lmdb, Time usage: 0.5620705989710595
Method: hdf5, Time usage: 0.28305170011717033

# Analisis 
Dalam eksperimen pengukuran waktu, waktu yang dibutuhkan untuk menyimpan satu gambar CIFAR-10 menggunakan metode penyimpanan yang berbeda (disk, LMDB, dan HDF5) diukur dengan menggunakan fungsi `timeit`. Hasilnya menunjukkan bahwa penyimpanan ke dalam format HDF5 membutuhkan waktu paling singkat, diikuti oleh penyimpanan ke dalam disk, dan terakhir penyimpanan ke dalam LMDB.

# Menyimpan banyak gambar

# penyesuaian kode
```python
store_many_disk(images, labels):
    """ Stores an array of images to disk
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)
    # Save all the images one by one
    for i, image in enumerate(images):
        Image.fromarray(image).save(disk_dir / f"{i}.png")
    # Save all the labels to the csv file
    with open(disk_dir / f"{num_images}.csv", "w") as csvfile:
        writer = csv.writer(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for label in labels:
            # This typically would be more than just one value per row
            writer.writerow([label])
def store_many_lmdb(images, labels):
    """ Stores an array of images to LMDB.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)
    map_size = num_images * images[0].nbytes * 10
    # Create a new LMDB DB for all the images
    env = lmdb.open(str(lmdb_dir / f"{num_images}_lmdb"), map_size=map_size)
    # Same as before — but let's write all the images in a single transaction
    with env.begin(write=True) as txn:
        for i in range(num_images):
            # All key-value pairs need to be Strings
            value = CIFAR_Image(images[i], labels[i])
            key = f"{i:08}"
            txn.put(key.encode("ascii"), pickle.dumps(value))
    env.close()
def store_many_hdf5(images, labels):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """
    num_images = len(images)
    # Create a new HDF5 file
    file = h5py.File(hdf5_dir / f"{num_images}_many.h5", "w")
    # Create a dataset in the file
    dataset = file.create_dataset(
        "images", np.shape(images), h5py.h5t.STD_U8BE, data=images
    )
    meta_set = file.create_dataset(
        "meta", np.shape(labels), h5py.h5t.STD_U8BE, data=labels
    )
    file.close()
```
# Analisis
Fungsi `store_many_disk` menyimpan array gambar ke dalam format disk dengan menyimpan setiap gambar sebagai file PNG terpisah dan menyimpan label-labelnya dalam sebuah file CSV terpisah. Fungsi `store_many_lmdb` mengonversi dan menyimpan array gambar dan label ke dalam format LMDB dengan menggunakan satu transaksi untuk menyimpan semua gambar dan label secara bersamaan. Fungsi `store_many_hdf5` menyimpan array gambar dan label ke dalam format HDF5 dengan membuat dataset untuk gambar-gambar dan label-labelnya dalam satu file.

# Dataset
```python
cutoffs = [10, 100, 1000, 10000, 100000]
# Let's double our images so that we have 100,000
images = np.concatenate((images, images), axis=0)
labels = np.concatenate((labels, labels), axis=0)
# Make sure you actually have 100,000 images and labels
print(np.shape(images))
print(np.shape(labels))
```
::: {.output .stream .stdout}
    (100000, 32, 32, 3)
    (100000,)

# Analisis
Dalam kode ini, `cutoffs` adalah daftar jumlah gambar yang akan diproses. Gambar-gambar dan label-labelnya akan digandakan sehingga menjadi 100.000. Hasil cetakan menampilkan dimensi array gambar dan label, memverifikasi bahwa jumlahnya telah mencapai 100.000.

# Storing many images
```python
_store_many_funcs = dict(
    disk=store_many_disk, lmdb=store_many_lmdb, hdf5=store_many_hdf5
)
from timeit import timeit
store_many_timings = {"disk": [], "lmdb": [], "hdf5": []}
for cutoff in cutoffs:
    for method in ("disk", "lmdb", "hdf5"):
        t = timeit(
            "_store_many_funcs[method](images_, labels_)",
            setup="images_=images[:cutoff]; labels_=labels[:cutoff]",
            number=1,
            globals=globals(),
        )
        store_many_timings[method].append(t)
        # Print out the method, cutoff, and elapsed time
        print(f"Method: {method}, Time usage: {t}")
```

# Analisis
Kode ini mengukur waktu eksekusi untuk menyimpan banyak gambar menggunakan metode penyimpanan yang berbeda (disk, LMDB, HDF5) untuk setiap jumlah yang ditentukan dalam `cutoffs`. Pengukuran dilakukan dengan menjalankan fungsi penyimpanan pada subset gambar dan label sejumlah `cutoff` sekali, menggunakan modul `timeit`. Hasilnya dicatat dalam `store_many_timings` untuk setiap metode penyimpanan.

# Membaca Single Image
# Dari Disk
```python
def read_single_disk(image_id):
    """ Stores a single image to disk.
        Parameters:
        ---------------
        image_id    integer unique ID for image
        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    image = np.array(Image.open(disk_dir / f"{image_id}.png"))
    with open(disk_dir / f"{image_id}.csv", "r") as csvfile:
        reader = csv.reader(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        label = int(next(reader)[0])
    return image, label
```

# ANalisis
Fungsi ini membaca sebuah gambar dari disk berdasarkan `image_id` yang diberikan. Gambar tersebut kemudian dimuat sebagai array menggunakan modul PIL dan label terkaitnya diambil dari file CSV yang sesuai. Kedua nilai ini kemudian dikembalikan sebagai tuple dari fungsi.

# membaca dari LMBD
```python
def read_single_lmdb(image_id):
    """ Stores a single image to LMDB.
        Parameters:
        ---------------
        image_id    integer unique ID for image
        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    # Open the LMDB environment
    env = lmdb.open(str(lmdb_dir / f"single_lmdb"), readonly=True)
    # Start a new read transaction
    with env.begin() as txn:
        # Encode the key the same way as we stored it
        data = txn.get(f"{image_id:08}".encode("ascii"))
        # Remember it's a CIFAR_Image object that is loaded
        cifar_image = pickle.loads(data)
        # Retrieve the relevant bits
        image = cifar_image.get_image()
        label = cifar_image.label
    env.close()
    return image, label
```

# Analisis
Fungsi ini membaca sebuah gambar dari basis data LMDB berdasarkan `image_id` yang diberikan. Lingkungan LMDB dibuka dalam mode hanya baca (readonly), lalu sebuah transaksi baca baru dimulai. Data yang sesuai dengan `image_id` diterjemahkan dari bentuk serial menggunakan pickle, kemudian array gambar dan labelnya diekstrak. Akhirnya, lingkungan ditutup dan nilai gambar beserta labelnya dikembalikan.

# Membaca dari HDFS
```python
def read_single_hdf5(image_id):
    """ Stores a single image to HDF5.
        Parameters:
        ---------------
        image_id    integer unique ID for image
        Returns:
        ----------
        image       image array, (32, 32, 3) to be stored
        label       associated meta data, int label
    """
    # Open the HDF5 file
    file = h5py.File(hdf5_dir / f"{image_id}.h5", "r+")
    image = np.array(file["/image"]).astype("uint8")
    label = int(np.array(file["/meta"]).astype("uint8"))
    return image, label
```

# Analisis 
Fungsi ini membaca sebuah gambar dari file HDF5 berdasarkan `image_id` yang diberikan. File HDF5 dibuka dalam mode pembacaan ("r+"), kemudian array gambar diekstrak dari dataset "image" dan labelnya dari dataset "meta". Nilai label dikonversi menjadi integer sebelum dikembalikan bersama dengan gambar.


# Fungsi membaca gambar
```python
_read_single_funcs = dict(
    disk=read_single_disk, lmdb=read_single_lmdb, hdf5=read_single_hdf5
)
```

# Analisis
Baris kode ini membuat kamus `_read_single_funcs` yang berisi fungsi-fungsi untuk membaca gambar dari tiga jenis penyimpanan yang berbeda: disk, LMDB, dan HDF5. Setiap kunci dalam kamus mengacu pada nama metode penyimpanan, sedangkan nilai-nilainya adalah fungsi-fungsi yang sesuai untuk membaca gambar dari setiap jenis penyimpanan.

# Membaca single Image
```python
from timeit import timeit
read_single_timings = dict()
for method in ("disk", "lmdb", "hdf5"):
    t = timeit(
        "_read_single_funcs[method](0)",
        setup="image=images[0]; label=labels[0]",
        number=1,
        globals=globals(),
    )
    read_single_timings[method] = t
    print(f"Method: {method}, Time usage: {t}")
```

# Analisis
Baris kode di atas mengukur waktu yang diperlukan untuk membaca sebuah gambar dari masing-masing jenis penyimpanan, yaitu disk, LMDB, dan HDF5. Pengukuran dilakukan menggunakan fungsi `timeit`, dengan menyediakan setup yang mencakup gambar pertama dari setiap jenis penyimpanan serta labelnya. Hasil waktu pembacaan kemudian disimpan dalam kamus `read_single_timings` dan dicetak untuk setiap metode penyimpanan.

# Membaca Banyak Gambar
# Kode awal
```python
def read_many_disk(num_images):
    """ Reads image from disk.
        Parameters:
        ---------------
        num_images   number of images to read
        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []
    # Loop over all IDs and read each image in one by one
    for image_id in range(num_images):
        images.append(np.array(Image.open(disk_dir / f"{image_id}.png")))
    with open(disk_dir / f"{num_images}.csv", "r") as csvfile:
        reader = csv.reader(
            csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        for row in reader:
            labels.append(int(row[0]))
    return images, labels
def read_many_lmdb(num_images):
    """ Reads image from LMDB.
        Parameters:
        ---------------
        num_images   number of images to read
        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []
    env = lmdb.open(str(lmdb_dir / f"{num_images}_lmdb"), readonly=True)
    # Start a new read transaction
    with env.begin() as txn:
        # Read all images in one single transaction, with one lock
        # We could split this up into multiple transactions if needed
        for image_id in range(num_images):
            data = txn.get(f"{image_id:08}".encode("ascii"))
            # Remember that it's a CIFAR_Image object
            # that is stored as the value
            cifar_image = pickle.loads(data)
            # Retrieve the relevant bits
            images.append(cifar_image.get_image())
            labels.append(cifar_image.label)
    env.close()
    return images, labels
def read_many_hdf5(num_images):
    """ Reads image from HDF5.
        Parameters:
        ---------------
        num_images   number of images to read
        Returns:
        ----------
        images      images array, (N, 32, 32, 3) to be stored
        labels      associated meta data, int label (N, 1)
    """
    images, labels = [], []
    # Open the HDF5 file
    file = h5py.File(hdf5_dir / f"{num_images}_many.h5", "r+")
    images = np.array(file["/images"]).astype("uint8")
    labels = np.array(file["/meta"]).astype("uint8")
    return images, labels
_read_many_funcs = dict(
    disk=read_many_disk, lmdb=read_many_lmdb, hdf5=read_many_hdf5
)
```

# Analisis
Fungsi-fungsi di atas adalah untuk membaca banyak gambar dari tiga jenis penyimpanan yang berbeda, yaitu disk, LMDB, dan HDF5. Setiap fungsi menerima parameter `num_images`, yang merupakan jumlah gambar yang akan dibaca, dan mengembalikan array gambar serta label yang sesuai. Kemudian, fungsi-fungsi ini digabungkan ke dalam kamus `_read_many_funcs` untuk penggunaan selanjutnya.

# Membaca banyak Gambar
```python
from timeit import timeit
read_many_timings = {"disk": [], "lmdb": [], "hdf5": []}
for cutoff in cutoffs:
    for method in ("disk", "lmdb", "hdf5"):
        t = timeit(
            "_read_many_funcs[method](num_images)",
            setup="num_images=cutoff",
            number=1,
            globals=globals(),
        )
        read_many_timings[method].append(t)
        # Print out the method, cutoff, and elapsed time
        print(f"Method: {method}, No. images: {cutoff}, Time usage: {t}")
```

# Analisis
Kode di atas mengukur waktu pembacaan sejumlah gambar dari setiap jenis penyimpanan (disk, LMDB, dan HDF5) untuk berbagai jumlah gambar yang berbeda, yang ditentukan oleh variabel `cutoffs`. Setiap waktu pembacaan disimpan dalam kamus `read_many_timings`. Hasilnya dicetak untuk setiap metode dan jumlah gambar yang dibaca. Dengan demikian, kita dapat membandingkan kinerja pembacaan antara berbagai jenis penyimpanan dan jumlah gambar.

# KESIMPULAN
Analisis kode yang dilakukan ini digunakan untuk memuat, menyimpan, dan membaca dataset CIFAR-10 dalam format yang berbeda, yaitu disk, LMDB, dan HDF5. kode ini juga digunakan untuk mengevaluasi kinerja penyimpanan dan pembacaan gambar tunggal serta banyak gambar dari setiap jenis penyimpanan. Dari hasil eksperimen, disimpulkan bahwa penyimpanan ke dalam format HDF5 membutuhkan waktu paling singkat, diikuti oleh penyimpanan ke dalam disk, dan terakhir penyimpanan ke dalam LMDB. Selain itu, pembacaan gambar tunggal membutuhkan waktu paling sedikit saat menggunakan format HDF5, sedangkan pembacaan banyak gambar tercepat terjadi pada format disk. Secara keseluruhan, pemilihan format penyimpanan tergantung pada kebutuhan spesifik aplikasi dan ketersediaan sumber daya.
