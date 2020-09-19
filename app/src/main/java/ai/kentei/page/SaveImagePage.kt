package ai.kentei.page

import android.Manifest
import android.content.ContentValues
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import androidx.core.app.ActivityCompat
import ai.kentei.R
import java.io.IOException
import java.util.*


class SaveImagePage : AppCompatActivity() {

    private val REQUEST_PERMISSION = 1000

    private var outputBmp: Bitmap? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView( R.layout.save_image_page)
        val myToolbar: Toolbar = findViewById(R.id.my_toolbar)
        setSupportActionBar(myToolbar)

        val outputString = intent.extras?.get(ChooseTypePage.OUTPUT) as String
        val outputBytes = Base64.getDecoder().decode(outputString)
        outputBmp = resizeBitmap(BitmapFactory.decodeByteArray(outputBytes, 0, outputBytes!!.size))

        val imageView = findViewById<ImageView>(R.id.outputImage)
        imageView.setImageBitmap(outputBmp)
    }

    private fun resizeBitmap(before: Bitmap): Bitmap {
        val height = ChooseTypePage.originalHeight
        val width  = ChooseTypePage.originalWidth

        return Bitmap.createScaledBitmap(before, width,  height,true)
    }

    fun clickSaveButton(view: View) {
        checkPermission();
    }

    fun clickRetryButton(view: View) {
        val intent = Intent(this@SaveImagePage, SelectImagePage::class.java)
        startActivity(intent)
    }

    private fun checkPermission() {
        // 既に許可している
        if (ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.WRITE_EXTERNAL_STORAGE
            )
            == PackageManager.PERMISSION_GRANTED
        ) {
            setUpWriteExternalStorage()
        } else {
            requestLocationPermission()
        }
    }

    private fun setUpWriteExternalStorage() {
        // 外部ストレージに画像を保存する
        try {
            val resolver = applicationContext.contentResolver

            val values = ContentValues().apply {
                put(MediaStore.Images.Media.TITLE, "ResumeFace.jpg")
                put(MediaStore.Images.Media.DISPLAY_NAME, "ResumeFace.jpg")
                put(MediaStore.Images.Media.MIME_TYPE, "image/jpg")
            }
            val url = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            Log.i("url: ", url.toString())
            if (url != null) resolver.openOutputStream(url).use { output ->
                Log.i("outputBmp:", outputBmp.toString())
                outputBmp?.compress(Bitmap.CompressFormat.JPEG, 50, output)
            }

            val toast = Toast.makeText(this, "画像が保存されました", Toast.LENGTH_SHORT)
            toast.show()

        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    private fun requestLocationPermission() {
        if (ActivityCompat.shouldShowRequestPermissionRationale(
                this,
                Manifest.permission.WRITE_EXTERNAL_STORAGE
            )
        ) {
            ActivityCompat.requestPermissions(
                this@SaveImagePage,
                arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE),
                REQUEST_PERMISSION
            )
        } else {
            val toast = Toast.makeText(this, "許可してください", Toast.LENGTH_SHORT)
            toast.show()
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE),
                REQUEST_PERMISSION
            )
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String?>, grantResults: IntArray
    ) {
        if (requestCode == REQUEST_PERMISSION) {
            // 使用が許可された
            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                setUpWriteExternalStorage()
            } else {
                // それでも拒否された時の対応
                val toast = Toast.makeText(this, "何もできません", Toast.LENGTH_SHORT)
                toast.show()
            }
        }
    }
}