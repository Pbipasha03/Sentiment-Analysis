import { Router, type IRouter } from "express";
import healthRouter from "./health";
import sentimentRouter from "./sentiment";

const router: IRouter = Router();

router.use(healthRouter);
router.use(sentimentRouter);

export default router;
