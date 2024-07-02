package com.example.entermatrix
import java.io.File
import android.net.Uri
import android.os.Bundle
import android.os.Build
import android.util.Log
import android.provider.MediaStore
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts

import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.layout.ContentScale

import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Bitmap
import android.graphics.Matrix

import android.content.ContentResolver
import android.content.Intent

import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Tensor

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private const val IMG_SIZE = 256

class MainActivity : ComponentActivity() {
    private var selectedImageUri by mutableStateOf<Uri?>(null)
    private var transformedImageBitmap by mutableStateOf<ImageBitmap?>(null)
    private var isModelLoaded by mutableStateOf(false)
    private lateinit var module: org.pytorch.Module

    private fun selectImageFromGallery() {
        val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
        launcher.launch(intent)
    }

    private val launcher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == RESULT_OK) {
                selectedImageUri = result.data?.data
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            enableEdgeToEdge()
        }

        try {
            Log.d("PytorchModel", "Начинаем загрузку модели...")
            module = LiteModuleLoader.load(assetFilePath("matrix_2.ptl"))
            Log.d("PytorchModel", "Модель загружена успешно")
            isModelLoaded = true
        } catch (e: Exception) {
            Log.e("PytorchModel", "Ошибка при загрузке модели: ${e.message}", e)  // Выводим стектрейс ошибки
        }
        setContent {
            MainScreen(
                selectedImageUri = selectedImageUri,
                onGalleryButtonClick = { selectImageFromGallery() },
                transformedImageBitmap = transformedImageBitmap,
                onTransformClick = {
                    if (isModelLoaded) {
                        transformImage()
                    } else {
                        // Модель еще не загружена, покажи сообщение
                    }
                },
                isModelLoaded = isModelLoaded

            )
        }
    }

    private fun transformImage() {
        selectedImageUri?.let { uri ->
            CoroutineScope(Dispatchers.Default).launch {
                try {
                    // Передаем contentResolver в decodeAndResizeBitmap
                    val bitmap = decodeAndResizeBitmap(uri, contentResolver)

                    if (bitmap != null) {
                        val inputTensor = bitmapToFloat32Tensor(bitmap)
                        Log.d("PytorchModel",  "Форма  входного  тензора:  ${inputTensor.shape().contentToString()}")
                        val inputTensorBatch = Tensor.fromBlob(
                            inputTensor.dataAsFloatArray,
                            longArrayOf(1, inputTensor.shape()[0], inputTensor.shape()[1], inputTensor.shape()[2])
                        )
                        val outputTensor = module.forward(IValue.from(inputTensorBatch)).toTensor()
                        //val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
                        Log.d("PytorchModel",  "Форма  выходного  тензора:  ${outputTensor.shape().contentToString()}")
                        val transformedBitmap = tensorToBitmap(outputTensor)

                        withContext(Dispatchers.Main) {
                            transformedImageBitmap = transformedBitmap.asImageBitmap()
                        }
                    }
                } catch (e: Exception) {
                    Log.e("PytorchModel", "Ошибка при обработке изображения: ${e.message}")
                }
            }
        }
    }

    private suspend fun decodeAndResizeBitmap(uri: Uri, contentResolver: ContentResolver): Bitmap? = withContext(Dispatchers.IO) {
        val bitmap = BitmapFactory.decodeStream(contentResolver.openInputStream(uri))
        bitmap?.let { Bitmap.createScaledBitmap(it, 256, 256, false) }
    }


    private fun assetFilePath(assetName: String): String {
        val file = File(filesDir, assetName)
        if (file.exists()) {
            return file.absolutePath
        }

        try {
            assets.open(assetName).use { inputStream ->
                file.outputStream().use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
        } catch (e: Exception) {
            Log.e("PytorchModel", "Ошибка при копировании asset: ${e.message}")
        }

        return file.absolutePath
    }
    private suspend fun bitmapToFloat32Tensor(bitmap: Bitmap): Tensor = withContext(Dispatchers.Default) {
        // 1. Изменение размера
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, IMG_SIZE, IMG_SIZE, true)

        // 2. Создание тензора и заполнение
        val floatArray = FloatArray(3 * IMG_SIZE * IMG_SIZE)
        val intArray = IntArray(IMG_SIZE * IMG_SIZE)
        resizedBitmap.getPixels(intArray, 0, IMG_SIZE, 0, 0, IMG_SIZE, IMG_SIZE)

        for (i in 0 until IMG_SIZE) {
            for (j in 0 until IMG_SIZE) {
                val pixel = intArray[i * IMG_SIZE + j]
                val index = (i * IMG_SIZE + j) * 3
                floatArray[index] = (Color.red(pixel)).toFloat() / 255.0f // Нормализация!
                floatArray[index + 1] = (Color.green(pixel)).toFloat() / 255.0f
                floatArray[index + 2] = (Color.blue(pixel)).toFloat() / 255.0f
            }
        }

        // 3. Создание тензора с нужным порядком осей
        val tensor = Tensor.fromBlob(floatArray, longArrayOf(3, IMG_SIZE.toLong(), IMG_SIZE.toLong()))

        tensor
    }

    private suspend fun tensorToBitmap(tensor: Tensor): Bitmap = withContext(Dispatchers.Default) {
        // 1. Денормализация
        val floatArray = tensor.dataAsFloatArray
        val pixels = IntArray(IMG_SIZE * IMG_SIZE)
        for (i in pixels.indices) {
            val r = (floatArray[i * 3] * 255.0f).toInt().coerceIn(0, 255)
            val g = (floatArray[i * 3 + 1] * 255.0f).toInt().coerceIn(0, 255)
            val b = (floatArray[i * 3 + 2] *  255.0f).toInt().coerceIn(0, 255)
            pixels[i] = Color.rgb(r, g, b)
        }

        // 2. Создание изображения
        val bitmap = Bitmap.createBitmap(IMG_SIZE, IMG_SIZE, Bitmap.Config.ARGB_8888)
        bitmap.setPixels(pixels, 0, IMG_SIZE, 0, 0, IMG_SIZE, IMG_SIZE)

        bitmap
    }

    @Composable
    fun MainScreen(
        selectedImageUri: Uri?,
        onGalleryButtonClick: () -> Unit,
        transformedImageBitmap: ImageBitmap?,
        onTransformClick: () -> Unit, isModelLoaded: Boolean
    ) {
        val context = LocalContext.current
        var selectedImage by remember { mutableStateOf<Bitmap?>(null) } // Состояние для выбранного Bitmap
        var isTransforming by remember { mutableStateOf(false) }  // Индикатор преобразования

        Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
            Column(
                modifier = Modifier
                    .padding(innerPadding)
                    .fillMaxSize(),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                Button(onClick = onGalleryButtonClick) {
                    Text("Выбрать из галереи")
                }

                Spacer(modifier = Modifier.height(16.dp))

                selectedImageUri?.let { nonNullUri -> // Добавлено let для Uri?
                    // Загружаем Bitmap только один раз после выбора изображения
                    LaunchedEffect(nonNullUri) {
                        selectedImage = withContext(Dispatchers.IO) {
                            decodeAndResizeBitmap(nonNullUri, context.contentResolver)
                        }
                    }
                    // Отображаем ProgressBar, пока изображение загружается
                    if (selectedImage == null) {
                        CircularProgressIndicator()
                    } else {
                        Image(
                            bitmap = selectedImage!!.asImageBitmap(),
                            contentDescription = "Выбранное изображение",
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                }

                Spacer(modifier = Modifier.height(16.dp))

                Button(onClick = onTransformClick, enabled = isModelLoaded) {
                    Text("Transform")
                }

                Spacer(modifier = Modifier.height(16.dp))

                transformedImageBitmap?.let { bitmap ->
                    Image(
                        bitmap = bitmap,
                        contentDescription = "Преобразованное изображение",
                        modifier = Modifier.fillMaxWidth(),
                        contentScale = ContentScale.Fit
                    )
                }
            }
        }
    }
}
