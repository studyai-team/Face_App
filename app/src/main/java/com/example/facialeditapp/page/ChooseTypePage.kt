package com.example.facialeditapp.page

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.ParcelFileDescriptor
import android.view.View
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import com.example.facialeditapp.Param
import com.example.facialeditapp.R
import com.example.facialeditapp.UploadImageHttpRequest
import java.io.FileDescriptor
import java.io.IOException


class ChooseTypePage : AppCompatActivity() {
    private val handler: Handler = Handler()
    private var imageUri: Uri = Uri.EMPTY

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.choose_type_page)
        val myToolbar: Toolbar = findViewById(R.id.my_toolbar)
        setSupportActionBar(myToolbar)

        imageUri = intent.extras?.get(SelectImagePage.IMAGE_URI) as Uri

        val imageView = findViewById<ImageView>(R.id.selectedImage)
        imageView.setImageURI(imageUri)

    }

    public fun clickEditButton(view: View) {
        try {
//            Thread(Runnable {
//                handler.post {
//                    val bmp: Bitmap = getBitmapFromUri(imageUri)
                    UploadImageHttpRequest(this).execute(
                        Param(
                            "http://192.168.10.6:9004", null
//                            bmp
                        )
                    )
//                }
//            }).start()
        } catch (e: IOException) {
            e.printStackTrace()
        }
//        val intent = Intent(this@ChooseTypePage, SaveImagePage::class.java)
//        startActivity(intent)
    }

//    @Throws(IOException::class)
//    private fun getBitmapFromUri(uri: Uri): Bitmap {
//        val parcelFileDescriptor: ParcelFileDescriptor? = contentResolver.openFileDescriptor(uri, "r")
//        val fileDescriptor: FileDescriptor? = parcelFileDescriptor?.fileDescriptor
//        val image: Bitmap = BitmapFactory.decodeFileDescriptor(fileDescriptor)
//        parcelFileDescriptor?.close()
//        return image
//    }
}