package com.example.hdy.controller;

import com.alibaba.fastjson.JSONObject;
import com.example.hdy.domain.Comment;
import com.example.hdy.service.impl.CommentServiceImpl;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;

import javax.servlet.http.HttpServletRequest;
import java.util.Date;

@RestController
@Controller
public class CommentController {
    @Autowired
    private CommentServiceImpl commentService;

//  提交评论
    @ResponseBody
    @RequestMapping(value = "/comment/add", method = RequestMethod.POST)
    public Object addComment(HttpServletRequest req){

        JSONObject jsonObject = new JSONObject();
        String userID = req.getParameter("userID");
        String videoID=req.getParameter("videoID");
        String content = req.getParameter("content").trim();

        Comment comment = new Comment();
        comment.setUserID(Integer.parseInt(userID));
        comment.setVideoID(Integer.parseInt(videoID));

        comment.setContent(content);
        comment.setCreateTime(new Date());
        boolean res = commentService.addComment(comment);
        if (res){
            jsonObject.put("code", 1);
            jsonObject.put("msg", "评论成功");
            return jsonObject;
        }else {
            jsonObject.put("code", 0);
            jsonObject.put("msg", "评论失败");
            return jsonObject;
        }
    }

//    获取所有评论列表
    @RequestMapping(value = "/comment", method = RequestMethod.GET)
    public Object allComment(){
        return commentService.allComment();
    }

//    获得指定videoID的评论列表
    @RequestMapping(value = "/comment/video/detail", method = RequestMethod.GET)
    public Object commentOfSongId(HttpServletRequest req){
        String videoID = req.getParameter("videoID");
        System.out.println("videoID:"+videoID);
        System.out.println(commentService.commentOfVideoID(Integer.parseInt(videoID)));
        return commentService.commentOfVideoID(Integer.parseInt(videoID));
    }


//    点赞
    @ResponseBody
    @RequestMapping(value = "/comment/like", method = RequestMethod.POST)
    public Object commentOfLike(HttpServletRequest req){

    JSONObject jsonObject = new JSONObject();
    String id = req.getParameter("id").trim();
    String up = req.getParameter("up").trim();

    Comment comment = new Comment();
    comment.setId(Integer.parseInt(id));
//    comment.setUp(Integer.parseInt(up));
    boolean res = commentService.updateCommentMsg(comment);
    if (res){
        jsonObject.put("code", 1);
        jsonObject.put("msg", "点赞成功");
        return jsonObject;
    }else {
        jsonObject.put("code", 0);
        jsonObject.put("msg", "点赞失败");
        return jsonObject;
    }
}

//    删除评论
    @RequestMapping(value = "/comment/delete", method = RequestMethod.GET)
    public Object deleteComment(HttpServletRequest req){
        String id = req.getParameter("id");
        return commentService.deleteComment(Integer.parseInt(id));
    }

//    更新评论
    @ResponseBody
    @RequestMapping(value = "/comment/update", method = RequestMethod.POST)
    public Object updateCommentMsg(HttpServletRequest req){
        JSONObject jsonObject = new JSONObject();
        String id = req.getParameter("id").trim();
        String userID = req.getParameter("userID").trim();
        String videoID = req.getParameter("videoID").trim();
        String content = req.getParameter("content").trim();

        Comment comment = new Comment();
        comment.setId(Integer.parseInt(id));
        comment.setUserID(Integer.parseInt(userID));
        if (videoID == "") {
            comment.setVideoID(0);
        } else {
            comment.setVideoID(Integer.parseInt(videoID));
        }

        comment.setContent(content);
//        comment.setUp(Integer.parseInt(up));

        boolean res = commentService.updateCommentMsg(comment);
        if (res){
            jsonObject.put("code", 1);
            jsonObject.put("msg", "修改成功");
            return jsonObject;
        }else {
            jsonObject.put("code", 0);
            jsonObject.put("msg", "修改失败");
            return jsonObject;
        }
    }
}
