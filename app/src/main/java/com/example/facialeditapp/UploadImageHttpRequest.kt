package com.example.facialeditapp

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.AsyncTask
import android.os.ParcelFileDescriptor
import android.view.View
import android.content.ContentResolver
import android.widget.TextView
import java.io.*
import java.net.HttpURLConnection
import java.net.URL


class UploadImageHttpRequest : AsyncTask<Param, Void, String> {
    private var mActivity: Activity? = null

    constructor(activity: Activity?) {
        mActivity = activity
    }

    override fun doInBackground(vararg params: Param): String? {
        val param: Param = params[0]
        var connection: HttpURLConnection? = null
        val sb = StringBuilder()
        try {
            // 画像をjpeg形式でstreamに保存
            val jpg = ByteArrayOutputStream()
            param.bmp?.compress(Bitmap.CompressFormat.JPEG, 100, jpg)
            val url = URL(param.uri)
            connection = url.openConnection() as HttpURLConnection
            connection.connectTimeout = 3000 //接続タイムアウトを設定する。
            connection.readTimeout = 3000 //レスポンスデータ読み取りタイムアウトを設定する。
            connection.requestMethod = "POST" //HTTPのメソッドをPOSTに設定する。
            //ヘッダーを設定する
            connection.setRequestProperty("User-Agent", "Android")
            connection.setRequestProperty("Content-Type", "application/octet-stream")
            connection.doInput = true //リクエストのボディ送信を許可する
            connection.doOutput = true //レスポンスのボディ受信を許可する
            connection.useCaches = false //キャッシュを使用しない
            connection.connect()

            // データを投げる
            val out: OutputStream = BufferedOutputStream(connection.outputStream)
            out.write(jpg.toByteArray())
            out.flush()

            // データを受け取る
            val `is`: InputStream = connection.inputStream
            val reader = BufferedReader(InputStreamReader(`is`, "UTF-8"))
            var line: String? = ""
            while (reader.readLine().also{line = it} != null) sb.append(line)
            `is`.close()
        } catch (e: IOException) {
            e.printStackTrace()
        } finally {
            connection?.disconnect()
        }
        return sb.toString()
    }

    override fun onPostExecute(string: String?) {
        // 戻り値をViewにセット
        val textView = mActivity!!.findViewById<View>(R.id.textView) as TextView
        textView.text = string
    }
}